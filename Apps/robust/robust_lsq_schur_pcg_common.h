#ifndef ROBUST_LSQ_COMMON_SCHUR_PCG_H
#define ROBUST_LSQ_COMMON_SCHUR_PCG_H

#include "robust_lsq_common.h"
#include "Math/v3d_linearbase.h"
#include "Math/v3d_nonlinlsq.h"
/* #include "Math/v3d_ldl_private.h" */
#include "utilities_common.h"
#include "Math/v3d_linear_lu.h"
#include "Math/v3d_linear_ldlt.h"
#include <iomanip>
#include <fstream>


using namespace std;
using namespace V3D;
namespace Robust_LSQ{

    struct Robust_LSQ_Optimizer_Schur_PCG_Base: public NLSQ_LM_Optimizer{ 

        static constexpr bool use_cheap_solver = 1;

#if defined(USE_ADAPTIVE_PCG_ITERATIONS)
        static constexpr int max_pcg_iterations_min = 1, max_pcg_iterations_max = 1024, max_pcg_iterations_std = 32;
        //static constexpr int max_pcg_iterations_min = 1, max_pcg_iterations_max = 1, max_pcg_iterations_std = 1;
#else
        static constexpr int max_pcg_iterations_std = 1000;
#endif

#if defined(USE_ADAPTIVE_PCG_ETA)
        static constexpr int n_eta_pcg_steps = 3; static constexpr double eta_pcg_vals[n_eta_pcg_steps] = { 0.1, 0.3, 0.5 };
        //static constexpr double eta_pcg_min = 1.0/16, eta_pcg_max = 1.0/2, eta_pcg_init = 1.0/4; // forcing sequence
        //static constexpr double eta_pcg_min = 0.1, eta_pcg_max = 0.4, eta_pcg_init = 0.2; // forcing sequence
        //static constexpr double eta_pcg_min = 0.1, eta_pcg_max = 0.529, eta_pcg_init = 0.23; // forcing sequence
        //static constexpr double eta_pcg_min = 0.1, eta_pcg_max = 0.5, eta_pcg_init = 0.25; // forcing sequence
        static constexpr int eta_pcg_init_step = 1;
#else
        static constexpr double eta_pcg_init = 0.1; // forcing sequence
#endif

        static constexpr double lambda_B_multiplier = 1.0;

        int max_pcg_iterations = 1000;

        Robust_LSQ_Optimizer_Schur_PCG_Base(NLSQ_ParamDesc const &paramDesc,
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions)
            : NLSQ_LM_Optimizer(paramDesc, costFunctions), _robustCostFunctions(robustCostFunctions),
            _nMeasurements(costFunctions[0]->_nMeasurements),
            _dimensionA(paramDesc.dimension[0]), _countA(paramDesc.count[0]),
            _dimensionB(paramDesc.dimension[1]), _countB(paramDesc.count[1]),
            _cur_countB(_countB),
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
            _deltaB(_countB, _dimensionB, &_DeltaB[0]),
            _M(_countA, _dimensionA, _dimensionA),
            _invM(_countA, _dimensionA, _dimensionA),
            _timer("BATimer"), _bestCost(1e20), _bestInlierRatio(0.0)
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

         template <typename VecArray>
         void multiply_A_v1(VecArray const& v, VectorArray<double>& Wt_v, VectorArray<double>& dst) const
         {
            for (int j = 0; j < _cur_countB; ++j) makeZeroVector(Wt_v[j]);
            for (int i = 0; i < _countA;     ++i) makeZeroVector(dst[i]);

            Vector<double> tmp(_dimensionB), tmp2(_dimensionA);

            for (int obj = 0; obj < _costFunctions.size(); obj++)
            {
                NLSQ_CostFunction const &costFun = *_costFunctions[obj];
                for (int k = 0; k < _cur_nMeasurements; ++k)
                {
                    int const i = costFun._correspondingParams[k][0];
                    int const j = costFun._correspondingParams[k][1];

                    if (j >= _cur_countB) continue;

                    multiply_At_v(_Ws[k], v[i], tmp); addVectorsIP(tmp, Wt_v[j]);
                } // end for (k)

                for (int j = 0; j < _cur_countB; ++j)
                {
                    int const start = _measurementStarts[j], count =_measurementCounts[j];

                    for (int p = 0; p < count; ++p)
                    {
                        int const k = _measurementIxs[start + p];
                        int const i = costFun._correspondingParams[k][0];
                        multiply_A_v(_W_invVs[k], Wt_v[j], tmp2); addVectorsIP(tmp2, dst[i]);
                    } // end for (p)
                } // end for (j)

                for (int i = 0; i < _countA; ++i)
                {
                    scaleVectorIP(-1.0, dst[i]);
                    multiply_A_v(_Us[i], v[i], tmp2); addVectorsIP(tmp2, dst[i]);
                } // end for (i)

            }
         } // end multiply_A_v()

        template <typename VecArray>
        void solve_M(VectorArray<double> const& r, VecArray &z) const
        {
            for (int i = 0; i < _countA; ++i) multiply_A_v(_invM[i], r[i], z[i]);
        }

        int run_PCG(VectorArrayAdapter<double>& x) const {

            if (use_cheap_solver)
            {
                this->solve_M(_Jt_eA_schur,x);
                return 1;
            }

            double const norm_g = norm_L2(_Jt_eA_schur);

            VectorArray<double> r(_countA, _dimensionA), z(_countA, _dimensionA), 
                p(_countA, _dimensionA), q(_countA, _dimensionA);

            VectorArray<double> work(_cur_countB, _dimensionB);

            for (int i = 0; i < _countA; ++i) makeZeroVector(x[i]);
            for (int i = 0; i < _countA; ++i) copyVector(_Jt_eA_schur[i], r[i]);
            this->solve_M(r, z);
            copyVectorArray(z, p);

            double r_dot_z = inner_product(r, z);

            for (int iter = 0; iter < max_pcg_iterations; ++iter)
            {
               this->multiply_A_v1(p, work, q);
               double const alpha = r_dot_z / inner_product(p, q);
               //cout << "|r|: " << norm_L2(r) << " |p|: " << norm_L2(p) << " |q|: " << norm_L2(q) << " |z|: " << norm_L2(z) << endl;

               add_scaled_vector_array_IP(alpha, p, x); add_scaled_vector_array_IP(-alpha, q, r);
               if (norm_L2(r) < _eta_pcg * norm_g)
               {
                  //cout << "Bipartite_NLSQ_LM_Optimizer_PCG::run_PCG(): leaving PCG after " << iter+1 << " iterations." << endl;
                  return iter+1;
               }

               this->solve_M(r, z);
               double const r_dot_z_new = inner_product(r, z), beta = r_dot_z_new / r_dot_z;
               r_dot_z = r_dot_z_new;
               for (int i = 0; i < _countA; ++i)
               {
                  scaleVectorIP(beta, p[i]); addVectorsIP(z[i], p[i]);
               }

               //cout << "iter: " << iter << " alpha: " << alpha << " beta: " << beta << endl;
            } // end for (iter)
            /* cout << "Bipartite_NLSQ_LM_Optimizer_PCG::run_PCG(): exhausted PCG iterations." << endl; */
            return max_pcg_iterations+1;


        }

         void init()
         {
            lambda = _damping;
            currentIteration = 0;
#if defined(USE_ADAPTIVE_PCG_ETA)
            _eta_pcg_step = eta_pcg_init_step;
            _eta_pcg      = eta_pcg_vals[eta_pcg_init_step];
#else
            _eta_pcg = eta_pcg_init;
#endif
            _prev_err  = -1;
            _delta_err = 0;
            /* this->fill_normal_equation(); */
         }

        void updateStat(double const &current_cost){
            _timer.stop();
            int const inls = count_inliers();
            double const inls_ratio = (double) inls/_nMeasurements;

            if (current_cost <= _bestCost){
                _bestCost = current_cost;
            }

            if (inls_ratio > _bestInlierRatio){
                _bestInlierRatio = inls_ratio;
            }

            double time = _timer.getTime();
            _stat_time.push_back(time);
            _stat_cost.push_back(_bestCost/_nMeasurements);
            _stat_inlier_ratio.push_back(_bestInlierRatio);
            _timer.start();
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

            _stat_time.clear(); _stat_cost.clear();
            _timer.start();

            for (int iter = 0; iter < maxIterations; iter++){

                this->preIterationCallback();

                double const lambda_multiplier = _cur_nMeasurements;
                double const lambda_eff   = lambda * lambda_multiplier;
                double const lambda_B_eff = lambda_B_multiplier * lambda_eff;
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
                updateStat(curr_err);

                double const norm_Linf_Jt_e = std::max(norm_Linf(_Jt_eA), norm_Linf(_Jt_eB));

                this->fillJacobians();
                this->evalJt_e(cached_weights);
                this->fillHessian(cached_weights);
                this->addDamping();
                this->prepare_schur();
                this->fillJt_e_schur();
                this->fill_M();
                /* this->fillJtJ_schur(); */

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

        int count_inliers()
        {
            NLSQ_CostFunction &costFun = *_costFunctions[0];
            int inls = 0;
            for (int i = 0; i < costFun._nMeasurements; ++i)
            {
                inls += costFun.isInlier(i);
            }
            return inls;
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
                    scaleMatrixIP(weights[k], J1tJ1); 
                    addMatricesIP(J1tJ1, _Us[i]);

                    //Point JtJ
                    multiply_At_A(Js2[k], J2tJ2); 
                    scaleMatrixIP(weights[k], J2tJ2);
                    addMatricesIP(J2tJ2, _Vs[j]);

                    //Camera vs Point JtJ
                    multiply_At_B(Js1[k], Js2[k], J1tJ2); 
                    scaleMatrixIP(weights[k], J1tJ2);
                    addMatricesIP(J1tJ2, _Ws[k]);
                }
            } //end for int obj
        }

        void prepare_schur(){
            //Schur complements
            
            Matrix<double> I(_dimensionB, _dimensionB); makeIdentityMatrix(I);
            for (int j = 0; j < _countB; j++){
                
                if (_measurementCounts[j] == 0){
                    continue;
                }
                LDLt<double> ldlt(_Vs[j]);
                ldlt.solveMat(I, _invVs[j]);
            } // end for j

            Matrix<double> W_invV_Wt(_dimensionA, _dimensionA);
            for (int obj = 0; obj < _costFunctions.size(); obj++){
                NLSQ_CostFunction const &costFun = *_costFunctions[obj];

                for (int k = 0; k < _nMeasurements; k++)
                {
                    int const j = costFun._correspondingParams[k][1];
                    multiply_A_B(_Ws[k], _invVs[j], _W_invVs[k]);
                }
                
            }//end for obj
        }

        void fillJt_e_schur()
        {
            for (int obj = 0; obj < _costFunctions.size(); obj++){

                NLSQ_CostFunction const &costFun = *_costFunctions[obj];
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

        //Fill the condition matrix
        void fill_M(){

            for (int i = 0; i < _countA; i++) 
                makeZeroMatrix(_M[i]);

            Matrix<double> W_invV_Wt(_dimensionA, _dimensionA);
            for (int obj = 0; obj < _costFunctions.size(); obj++){
                NLSQ_CostFunction const &costFun = *_costFunctions[obj];
                for (int j = 0; j < _countB; j++){
                    int const start = _measurementStarts[j], count = _measurementCounts[j];

                    for (int p = 0; p < count; p++){
                        int const k = _measurementIxs[start + p];
                        int const i = costFun._correspondingParams[k][0];
                        multiply_A_Bt(_W_invVs[k], _Ws[k], W_invV_Wt);
                        addMatricesIP(W_invV_Wt, _M[i]);
                    }
                }
            }

            Matrix<double> I(_dimensionA, _dimensionA); makeIdentityMatrix(I);
            for (int i = 0; i < _countA; ++i)
            {
               scaleMatrixIP(-1.0, _M[i]); addMatricesIP(_Us[i], _M[i]);
               LDLt<double> ldlt(_M[i]);
               ldlt.solveMat(I, _invM[i]);
            }
        }

        void evalJt_e(vector<Vector<double>> const &given_weights) 
        {
            for (int obj = 0; obj < _costFunctions.size(); obj++){
                MatrixArray<double> const &Js1 = *_residuals[obj]->_Js[0];
                MatrixArray<double> const &Js2 = *_residuals[obj]->_Js[1];
                auto const &residuals = _residuals[obj]->_residuals;
                auto const &costFun = *_costFunctions[obj];

                Vector<double> Jt_eA(_dimensionA), Jt_eB(_dimensionB);
                for (int i = 0; i < _countA; i++) makeZeroVector(_Jt_eA[i]);
                for (int j = 0; j < _countB; j++) makeZeroVector(_Jt_eB[j]);

                for (int k = 0; k < _nMeasurements; k++){
                    int const i = costFun._correspondingParams[k][0];
                    int const j = costFun._correspondingParams[k][1];

                    multiply_At_v(Js1[k], residuals[k], Jt_eA);
                    multiply_At_v(Js2[k], residuals[k], Jt_eB);

                    scaleVectorIP(given_weights[obj][k], Jt_eA);
                    scaleVectorIP(given_weights[obj][k], Jt_eB);

                    addVectorsIP(Jt_eA, _Jt_eA[i]);
                    addVectorsIP(Jt_eB, _Jt_eB[j]);
                }
            }
        }

       
        void solveJtJ()
        {

            int n_pcg_iterations = 0;
            {
               // Solve the system
               n_pcg_iterations = this->run_PCG(_deltaA);
               for (int j = 0; j < _countB; ++j) makeZeroVector(_Jt_eB_schur[j]);

               Vector<double> Wt_deltaA(_dimensionB), rhs(_dimensionB);

               for (int obj = 0; obj < _costFunctions.size(); obj++){
                NLSQ_CostFunction const &costFun = *_costFunctions[obj];
                for (int k = 0; k < _nMeasurements; ++k)
                {
                    int const i = costFun._correspondingParams[k][0];
                    int const j = costFun._correspondingParams[k][1];

                    /* if (j >= _cur_countB) continue; */

                    multiply_At_v(_Ws[k], _deltaA[i], Wt_deltaA);
                    addVectorsIP(Wt_deltaA, _Jt_eB_schur[j]);
                } // end for (k)
               }

               for (int j = 0; j < _countB; ++j)
               {
                  if (_measurementCounts[j] > 0)
                  {
                     addVectors(_Jt_eB_schur[j], _Jt_eB[j], rhs);
                     multiply_At_v(_invVs[j], rhs, _deltaB[j]);
                     //LDLt<double> chol(_Vs[j]); copyVector(chol.solveVec(rhs), deltaB[j]);
                  }
                  else
                     makeZeroVector(_deltaB[j]);
               } // end for (j)
               scaleVectorIP(-1.0, _DeltaB);
               // cout << "|deltaA| = " << norm_L2(DeltaA) << " |deltaB| = " << norm_L2(DeltaB) << endl;
               // cout << "|Jt_eA| = " << norm_L2(_Jt_eA) << " |Jt_eA_schur| = " << norm_L2(_Jt_eA_schur) << endl;
            } // end scope

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

        //Store running statistics
        std::vector<double> _stat_time, _stat_cost, _stat_inlier_ratio;
        V3D::Timer _timer;
        double _bestCost, _bestInlierRatio;

        protected:
        int const _dimensionA, _countA;
        int const _dimensionB, _countB;
        int const _totalParamDimension;
        int const _nMeasurements;
        int _cur_countB, _cur_nMeasurements;
        vector<int> _measurementStarts, _measurementCounts, _measurementIxs;
        Vector<double> _DeltaA, _DeltaB;
        VectorArrayAdapter<double> _deltaA, _deltaB;

        Matrix<double> _JtJ;
        MatrixArray<double> _Us, _Vs, _invVs, _Ws, _W_invVs, _JtJ_blocks, _M, _invM;
        VectorArray<double> _Jt_eA, _Jt_eB, _Jt_eA_schur, _Jt_eB_schur;

        double _curr_err, _prev_err, _delta_err, _eta_pcg;


    };

}//end namespace Robust_LSQ

#endif
