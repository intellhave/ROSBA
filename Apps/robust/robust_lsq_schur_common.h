#ifndef ROBUST_LSQ_COMMON_SCHUR_H
#define ROBUST_LSQ_COMMON_SCHUR_H

#include "robust_lsq_common.h"
#include "Math/v3d_nonlinlsq.h"
/* #include "Math/v3d_ldl_private.h" */
/* #include "utilities_common.h" */
#include "Math/v3d_linear_lu.h"
#include "Math/v3d_linear_ldlt.h"
#include <iomanip>
#include <fstream>


enum SOLVER_METHOD
{
    USE_GRADIENT_SOLVER = 0,
    USE_DIAGONAL_SOLVER = 1,
    USE_BLOCK_DIAGONAL_SOLVER = 2,
    USE_CHOLESKY_SOLVER = 3
};
constexpr int solver_method = 3;

using namespace std;
using namespace V3D;
namespace Robust_LSQ{

    struct Robust_LSQ_Optimizer_Schur_Base: public NLSQ_LM_Optimizer{ 
        Robust_LSQ_Optimizer_Schur_Base(NLSQ_ParamDesc const &paramDesc,
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions)
            : NLSQ_LM_Optimizer(paramDesc, costFunctions), _robustCostFunctions(robustCostFunctions),
            _nMeasurements(costFunctions[0]->_nMeasurements),
            _dimensionA(paramDesc.dimension[0]), _countA(paramDesc.count[0]),
            _dimensionB(paramDesc.dimension[1]), _countB(paramDesc.count[1]),
            _totalParamDimension(_countA * _dimensionA + _countB * _dimensionB),
            _measurementCounts(_countB), _measurementStarts(_countB),
            _measurementIxs(costFunctions[0]->_nMeasurements),
            _JtJ(_dimensionA * _countA, _dimensionA * _countA),
            _Us(_countA, _dimensionA, _dimensionA),
            _Vs(_countB, _dimensionB, _dimensionB),
            _invVs(_countB, _dimensionB, _dimensionB),
            _Ws(costFunctions[0]->_nMeasurements, _dimensionA, _dimensionB),
            _W_invVs(costFunctions[0]->_nMeasurements, _dimensionA, _dimensionB),
            _JtJ_blocks(_countA * _countA, _dimensionA, _dimensionA),
            _Jt_eA(_countA, _dimensionA), _Jt_eB(_countB, _dimensionB),
            _Jt_eA_schur(_countA, _dimensionA), _Jt_eB_schur(_countB, _dimensionB),
            _DeltaA(_countA * _dimensionA), _DeltaB(_countB * _dimensionB),
            _deltaA(_countA, _dimensionA, &_DeltaA[0]),
            _deltaB(_countB, _dimensionB, &_DeltaB[0])
        {
            //This type of optimizer only receives bipartite problems
            assert(paramDesc.nParamTypes==2);
            NLSQ_CostFunction const &costFun = *costFunctions[0];
            
            //Handle the mapping from point (B) to cameras (A);
            fillVector(0, _measurementCounts);
            for (int k = 0; k < _nMeasurements; k++){
                int const j = costFun._correspondingParams[k][1];
                ++_measurementCounts[j];
            }

            _measurementStarts[0] = 0;
            for (int j = 1; j < _countB; j++) 
                _measurementStarts[j] = _measurementStarts[j-1] + _measurementCounts[j-1];

            fillVector(0, _measurementCounts);
            for (int k = 0; k < _nMeasurements; k++){
                int const j = costFun._correspondingParams[k][1];
                int const pos = _measurementStarts[j] + _measurementCounts[j];
                _measurementIxs[pos] = k;
                ++_measurementCounts[j];
            }

            //Fill weights of 1.0 to residual weights;
            for (int obj = 0; obj < costFunctions.size(); obj++){
                fillVector(1.0, _residuals[obj]->_weights);
            }
            makeZeroVector(_DeltaA);
            makeZeroVector(_DeltaB);
        }

        void minimize()
        {
            status = LEVENBERG_OPTIMIZER_TIMEOUT;
            bool computeDerivatives = true;
            int const nObjs = _costFunctions.size();
            vector<Vector<double>> cached_errors(nObjs);
            vector<Vector<double>> cached_weights(nObjs);

            for (int obj = 0; obj < nObjs; obj++){
                cached_errors[obj].newsize(_nMeasurements);
                cached_weights[obj].newsize(_nMeasurements);
                fillVector(1.0, cached_weights[obj]);
            }

            int const totalParamDimension = _countA*_dimensionA + _countB*_dimensionB;
            assert(totalParamDimension > 0);

            Vector<double> DeltaA(_countA * _dimensionA), DeltaB(_countB * _dimensionB);
            VectorArrayAdapter<double> deltaA(_countA, _dimensionA, &DeltaA[0]);
            VectorArrayAdapter<double> deltaB(_countB, _dimensionB, &DeltaB[0]);

            double curr_err = 0.0;
            double cur_lambda[64];

           

            for (int iter = 0; iter < maxIterations; iter++){

                this->preIterationCallback();
                for (int obj = 0; obj < nObjs; obj++){

                    NLSQ_CostFunction const &costFun = *_costFunctions[obj];
                    Robust_NLSQ_CostFunction_Base const &robustCostFun = *_robustCostFunctions[obj];
                    auto &residuals = *_residuals[obj];
                    //compute residuals
                    robustCostFun.cache_residuals(residuals, cached_errors[obj]);
                    //Cache weights (due to robust kernels)
                    robustCostFun.cache_weight_fun(1.0, cached_errors[obj], cached_weights[obj]);
                }

                /* curr_err = eval_current_cost(cached_errors); */
                curr_err = eval_current_cost(cached_errors);
                std::cout << "It: " << iter << " err : " << curr_err << " rep err : " 
                    << sqrt(curr_err)/_nMeasurements << " lambda : " << _damping
                    << std::endl;

                this->fillJacobians();
                this->evalJt_e(cached_weights);
                this->fillHessian(cached_weights);
                this->addDamping();
                this->fillJtJ_schur();

                this->solveJtJ();
                //Update parameters
                this->saveAllParameters();

                this->updateParameters(0, _deltaA);
                this->updateParameters(1, _deltaB);
                this->finishUpdateParameters();

                //Evaluate new cost
                for (int obj = 0; obj < nObjs; obj++){
                    Robust_NLSQ_CostFunction_Base const &robustCostFun = *_robustCostFunctions[obj];
                    robustCostFun.cache_residuals(*_residuals[obj], cached_errors[obj]);
                }
                double const new_err = eval_current_cost(cached_errors);


                if (new_err < curr_err){
                    _damping = max(1e-8, _damping * 0.1);
                } else {
                    this->restoreAllParameters();
                    _damping = _damping * 10;
                }
            }//end for iteration


        }//end minimize


        //Evaluation of total robust cost.
        double eval_current_cost (vector<Vector<double>> const &cached_errors) const{
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; obj++){
                NLSQ_CostFunction const &costFun =*_costFunctions[obj];
                auto const &residuals = _residuals[obj];
                /* for (int k = 0; k < cached_errors[obj].size(); k++) */
                /*     cost += cached_errors[obj][k]; */
                cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            }
            return cost;
        }


        void addDamping(double const w = 1.0){
               // Augment the diagonals
               for (int i = 0; i < _countA; ++i)
                  for (int l = 0; l < _dimensionA; ++l) _Us[i][l][l] += w * _damping;
               for (int j = 0; j < _countB; ++j)
                  for (int l = 0; l < _dimensionB; ++l) _Vs[j][l][l] += w * _damping;
        }

        void fillHessian(vector<Vector<double>> const &given_weights){

            Matrix<double> J1tJ1(_dimensionA, _dimensionA), 
                J2tJ2(_dimensionB, _dimensionB), J1tJ2(_dimensionA, _dimensionB);

            for (int i = 0; i < _countA; ++i) makeZeroMatrix(_Us[i]);
            for (int j = 0; j < _countB; ++j) makeZeroMatrix(_Vs[j]);
            for (int k = 0; k < _nMeasurements; ++k) makeZeroMatrix(_Ws[k]);

            for (int obj = 0; obj < _costFunctions.size(); obj++){
                MatrixArray<double> const &Js1 = *_residuals[obj]->_Js[0];
                MatrixArray<double> const &Js2 = *_residuals[obj]->_Js[1];
                auto const &residuals = _residuals[obj]->_residuals;
                auto const &costFun = *_costFunctions[obj];
                Vector<double> const &weights = given_weights[obj];

                for (int k = 0; k < _nMeasurements; k++){
                    int const i = costFun._correspondingParams[k][0];
                    int const j = costFun._correspondingParams[k][1];

                    //Camera JtJ
                    multiply_At_A(Js1[k], J1tJ1); 
                    scaleMatrixIP(1.0 * weights[k], J1tJ1); 
                    addMatricesIP(J1tJ1, _Us[i]);

                    //Point JtJ
                    multiply_At_A(Js2[k], J2tJ2); 
                    scaleMatrixIP(1.0 * weights[k], J2tJ2);
                    addMatricesIP(J2tJ2, _Vs[j]);

                    //Camera vs Point JtJ
                    multiply_At_B(Js1[k], Js2[k], J1tJ2); 
                    scaleMatrixIP(1.0 * weights[k], J1tJ2);
                    addMatricesIP(J1tJ2, _Ws[k]);
                }
            }
        }

        void fillJtJ_schur(){
            //Schur complements
            
            Matrix<double> I(_dimensionB, _dimensionB); makeIdentityMatrix(I);
            for (int j = 0; j < _countB; j++){
                
                if (_measurementCounts[j] == 0){
                    continue;
                }
                LDLt<double> ldlt(_Vs[j]);
                ldlt.solveMat(I, _invVs[j]);
            }

            Matrix<double> W_invV_Wt(_dimensionA, _dimensionA);
            for (int obj = 0; obj < _costFunctions.size(); obj++){
                NLSQ_CostFunction const &costFun = *_costFunctions[obj];

                for (int k = 0; k < _nMeasurements; k++)
                {
                    int const j = costFun._correspondingParams[k][1];
                    multiply_A_B(_Ws[k], _invVs[j], _W_invVs[k]);
                }

                if (solver_method > USE_GRADIENT_SOLVER){

                    if (solver_method >= USE_CHOLESKY_SOLVER)
                    {
                        for (int p = 0; p < _JtJ_blocks.count(); ++p) makeZeroMatrix(_JtJ_blocks[p]);
                    }
                    else
                    {
                        for (int i = 0; i < _countA; ++i) makeZeroMatrix(_JtJ_blocks[i*(_countA+1)]);
                    }

                    for (int j = 0; j < _countB; ++j){
                        int const start = _measurementStarts[j], count = _measurementCounts[j];

                        switch (solver_method){
                            case USE_DIAGONAL_SOLVER:
                                for (int p = 0; p < count ; ++p){
                                    int const k = _measurementIxs[start + p];
                                    int const i = costFun._correspondingParams[k][0];

                                    auto &C = _JtJ_blocks[i + _countA*i];
                                    auto const& W_invV = _W_invVs[k];
                                    auto const& W      = _Ws[k];

                                    for (int l1 = 0; l1 < _dimensionA; ++l1)
                                    {
                                        double sum = 0;
                                        for (int l2 = 0; l2 < _dimensionB; ++l2) sum += W_invV[l1][l2] * W[l1][l2];
                                        C[l1][l1] += sum;
                                    }

                                }
                                break;
                            case USE_BLOCK_DIAGONAL_SOLVER:
                                for (int p = 0; p < count; p++){
                                    int const k = _measurementIxs[start + p];
                                    int const i = costFun._correspondingParams[k][0];
                                    multiply_A_Bt(_W_invVs[k], _Ws[k], W_invV_Wt);
                                    addMatricesIP(W_invV_Wt, _JtJ_blocks[i + _countA * i]);
                                }
                                break;

                           default: 
                                for (int p1 = 0; p1 < count; p1++)
                                    for (int p2 = 0; p2 < count; p2++){
                                        int const k1 = _measurementIxs[start + p1];
                                        int const k2 = _measurementIxs[start + p2];

                                        int const i1 = costFun._correspondingParams[k1][0];
                                        int const i2 = costFun._correspondingParams[k2][0];

                                        multiply_A_Bt(_W_invVs[k1], _Ws[k2], W_invV_Wt);
                                        addMatricesIP(W_invV_Wt, _JtJ_blocks[i1 + _countA * i2]);
                                    }

                        } //end switch solver method
                    } //end for j

                    switch (solver_method)
                    {
                        case USE_DIAGONAL_SOLVER:
                        case USE_BLOCK_DIAGONAL_SOLVER:
                            for (int i = 0; i < _countA; ++i) 
                                scaleMatrixIP(-1.0, _JtJ_blocks[i + _countA*i]);
                            break;
                        default: //CHOLESKY SOLVER
                            for (int p = 0; p < _JtJ_blocks.count(); ++p) 
                                scaleMatrixIP(-1.0, _JtJ_blocks[p]);
                    } // end switch (solver_method)

                    for (int i = 0; i < _countA; ++i) 
                        addMatricesIP(_Us[i], _JtJ_blocks[i + _countA*i]);

                } //end if solver > GRADIENT SOLVER

                //Now, handle the r.h.s
                for (int i = 0; i < _countA; i++)
                    makeZeroVector(_Jt_eA_schur[i]);

                Vector<double> W_invV_b(_dimensionA);
                for (int j = 0; j < _countB; j++){
                    int const start = _measurementStarts[j], count = _measurementCounts[j];
                    for (int p = 0; p < count; p++){
                        int const k = _measurementIxs[start + p];
                        int const i = costFun._correspondingParams[k][0];
                        multiply_A_v(_W_invVs[k], _Jt_eB[j], W_invV_b);
                        addVectorsIP(W_invV_b, _Jt_eA_schur[i]);
                    }//end for p
                } //end for j 
                for (int i = 0; i < _countA; i++){
                    subtractVectors(_Jt_eA_schur[i], _Jt_eA[i], _Jt_eA_schur[i]);
                }

            }//end for obj
        }

        void evalJt_e(vector<Vector<double>> const &given_weights) 
        {
            Vector<double> Jt_eA(_dimensionA), Jt_eB(_dimensionB);
            for (int i = 0; i < _countA; i++) makeZeroVector(_Jt_eA[i]);
            for (int j = 0; j < _countB; j++) makeZeroVector(_Jt_eB[j]);

            for (int obj = 0; obj < _costFunctions.size(); obj++){
                MatrixArray<double> const &Js1 = *_residuals[obj]->_Js[0];
                MatrixArray<double> const &Js2 = *_residuals[obj]->_Js[1];
                auto const &residuals = _residuals[obj]->_residuals;
                auto const &costFun = *_costFunctions[obj];


                for (int k = 0; k < _nMeasurements; k++){
                    int const i = costFun._correspondingParams[k][0];
                    int const j = costFun._correspondingParams[k][1];

                    multiply_At_v(Js1[k], residuals[k], Jt_eA);
                    multiply_At_v(Js2[k], residuals[k], Jt_eB);

                    scaleVectorIP(1.0 * given_weights[obj][k], Jt_eA);
                    scaleVectorIP(1.0 * given_weights[obj][k], Jt_eB);

                    addVectorsIP(Jt_eA, _Jt_eA[i]);
                    addVectorsIP(Jt_eB, _Jt_eB[j]);
                }
            }
        }

        void solveJtJ()
        {

            switch (solver_method){
                case USE_GRADIENT_SOLVER:
                    {
                        double const rcp_lambda = 1.0/_damping;
                        for (int i = 0; i < _countA; i++)
                        {
                            for (int l = 0; l < _dimensionA; ++l) 
                                _deltaA[i][l] = rcp_lambda * _Jt_eA_schur[i][l];
                        }
                        break;
                    }
                case USE_DIAGONAL_SOLVER:
                    for (int i = 0; i < _countA; ++i)
                    {
                        auto const& Ai = _JtJ_blocks[i + _countA*i];
                        for (int l = 0; l < _dimensionA; ++l) 
                            _deltaA[i][l] = _Jt_eA_schur[i][l] / Ai[l][l];
                    } // end for (i)
                    break;
                case USE_BLOCK_DIAGONAL_SOLVER:
                    // We apply a cheap approximation for now to solve for deltaA
                    for (int i = 0; i < _countA; ++i)
                    {
                        LDLt<double> chol(_JtJ_blocks[i + _countA*i]);
                        copyVector(chol.solveVec(_Jt_eA_schur[i]), _deltaA[i]);
                    } // end for (i)
                    break;

                case USE_CHOLESKY_SOLVER:
                    {
                        for (int i1 = 0; i1 < _countA; ++i1)
                            for (int i2 = 0; i2 < _countA; ++i2)
                                copyMatrixSlice(_JtJ_blocks[i1 + _countA*i2], 0, 0, _dimensionA, _dimensionA, _JtJ, i1*_dimensionA, i2*_dimensionA);
                        LDLt<double> chol(_JtJ);
                        Vector<double> a(_countA * _dimensionA, &_Jt_eA_schur[0][0]);
                        copyVector(chol.solveVec(a), _DeltaA);
                    }
            } //end switch(solver_method)


            for (int j = 0; j < _countB; ++j) makeZeroVector(_Jt_eB_schur[j]);
            Vector<double> Wt_deltaA(_dimensionB), rhs(_dimensionB);

            for (int obj = 0; obj < _costFunctions.size(); obj++){
                NLSQ_CostFunction const &costFun = *_costFunctions[obj];
                for (int k = 0; k < _nMeasurements; ++k)
                {
                    int const i = costFun._correspondingParams[k][0];
                    int const j = costFun._correspondingParams[k][1];

                    multiply_At_v(_Ws[k], _deltaA[i], Wt_deltaA);
                    addVectorsIP(Wt_deltaA, _Jt_eB_schur[j]);
                } // end for (k)
            }

            for (int j = 0; j < _countB; ++j)
            {
                addVectors(_Jt_eB_schur[j], _Jt_eB[j], rhs);
                multiply_At_v(_invVs[j], rhs, _deltaB[j]);
                //LDLt<double> chol(_Vs[j]); copyVector(chol.solveVec(rhs), deltaB[j]);
            } // end for (j)
            scaleVectorIP(-1.0, _DeltaB);
        }



        std::vector<Robust_NLSQ_CostFunction_Base *> const &_robustCostFunctions;
        double _damping = 1e-3;

        protected:
        int const _dimensionA, _countA;
        int const _dimensionB, _countB;
        int const _totalParamDimension;
        int const _nMeasurements;
        vector<int> _measurementStarts, _measurementCounts, _measurementIxs;
        Vector<double> _DeltaA, _DeltaB;
        VectorArrayAdapter<double> _deltaA, _deltaB;

        Matrix<double> _JtJ;
        MatrixArray<double> _Us, _Vs, _invVs, _Ws, _W_invVs, _JtJ_blocks;
        VectorArray<double> _Jt_eA, _Jt_eB, _Jt_eA_schur, _Jt_eB_schur;


    };

}//end namespace Robust_LSQ
#endif

