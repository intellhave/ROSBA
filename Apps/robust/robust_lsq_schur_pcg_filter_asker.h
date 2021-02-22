#ifndef ROBUST_LSQ_SCHUR_PCG_FILTER_ASKER_H
#define ROBUST_LSQ_SCHUR_PCG_FILTER_ASKER_H 
#include "robust_lsq_common.h"
#include "Math/v3d_linearbase.h"
#include "Math/v3d_nonlinlsq.h"
#include "utilities_common.h"
#include "Math/v3d_linear_lu.h"
#include "Math/v3d_linear_ldlt.h"

#include "robust_lsq_schur_pcg_common.h"
#include "robust_lsq_lifted.h"
#include <iomanip>
#include <fstream>
#include <random>

#include "filter.h"

typedef InlineMatrix<double, 2, 1> Matrix2x1d;
typedef InlineVector<double, 1> Vector1d;

namespace Robust_LSQ{

   struct Robust_LSQ_Optimizer_Schur_PCG_Filter_Asker : public Robust_LSQ_Optimizer_Schur_PCG_Base{

        typedef Robust_LSQ_Optimizer_Schur_PCG_Base Base;
        Robust_LSQ_Optimizer_Schur_PCG_Filter_Asker(NLSQ_ParamDesc const &paramDesc,     
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions): 
            Base(paramDesc, costFunctions, robustCostFunctions), 
            _s(costFunctions[0]->_nMeasurements),
            _storeS(costFunctions[0]->_nMeasurements),
            _gradS(costFunctions[0]->_nMeasurements),
            _scaledErrors(costFunctions[0]->_nMeasurements),
            _scaledResiduals(costFunctions[0]->_nMeasurements)
        {
            _nMeasurements = _costFunctions[0]->_nMeasurements;
            makeZeroVector(_s); 
            makeZeroVector(_gradS); 
            makeZeroVector(_scaledErrors); 
        }

        enum FilterStep { MOO, EXPLORATION, };
        void storeS(){ for (int k = 0; k < _nMeasurements; k++){ _storeS[k] = _s[k];}}
        void restoreS(){ for (int k = 0; k < _nMeasurements; k++){ _s[k] = _storeS[k];}}

        Matrix2x1d JacobianS (double const s, int const k) const {
            NLSQ_Residuals &residuals = *_residuals[0];
            
            double  const w = s / (sqrt(1 + s*s));
            Matrix2x1d J; 
            J[0][0] = w * residuals._residuals[k][0];
            J[1][0] = w * residuals._residuals[k][1];
            return J;
        }

        double computeScale (double const &s) const {
            return 1.0/sqrt(1.0 + sqr(s));
        }

        double inverseWeight(Matrix2x1d const &Js) const {
            double sJtJ = _mu * (sqr(Js[0][0]) + sqr(Js[1][0])) 
                + (_mu1 +  damping_constraints + damping_value) ; 
            return 1.0/sJtJ;
        }

        void evalFH(double &f, double &h){
            f = 0; h = 0;
            NLSQ_CostFunction &costFun = *_costFunctions[0];
            NLSQ_Residuals& residuals = *_residuals[0];
            Robust_NLSQ_CostFunction_Base &robustCost = *_robustCostFunctions[0];
            for (int k = 0; k < _nMeasurements; k++){
                double const sk = sqrt(1.0 + sqr(_s[k]));
                _scaledResiduals[k][0] = residuals._residuals[k][0]/sk;
                _scaledResiduals[k][1] = residuals._residuals[k][1]/sk;
                _scaledErrors[k] = sqrNorm_L2(_scaledResiduals[k]);
                h += sqr(_s[k]);
            }
            f+= robustCost.eval_target_cost(1.0, _scaledErrors);
        }

        void scaleJacobians()
        {
            NLSQ_CostFunction &costFun = *_costFunctions[0];
            NLSQ_Residuals &residuals = *_residuals[0];
            Robust_NLSQ_CostFunction_Base &robustCostFun = *_robustCostFunctions[0];

            int const nParamTypes = costFun._usedParamTypes.size();
            int const nMeasurements = costFun._nMeasurements;

            for (int i = 0; i < nParamTypes; i++){
                int const paramType = costFun._usedParamTypes[i];
                int const paramDim = _paramDesc.dimension[paramType];

                MatrixArray<double>  &J = *residuals._Js[i];

                for (int k = 0; k < nMeasurements; k++){
                    double const sk = computeScale(_s[k]);
                    scaleMatrixIP(sk, J[k]);
                }
            }
        }

        void evalJt_e(Vector<double> const &weights, FilterStep const &filterStep = FilterStep::MOO){

            makeZeroVector(_gradS);
            NLSQ_CostFunction &costFun = *_costFunctions[0];
            Robust_NLSQ_CostFunction_Base &robustCostFun = *_robustCostFunctions[0];
            int const nMeasurements = costFun._nMeasurements;

            MatrixArray<double> const &Js1 = *_residuals[0]->_Js[0];
            MatrixArray<double> const &Js2 = *_residuals[0]->_Js[1];
            auto const &residuals = _residuals[0]->_residuals;
            Vector<double> Jt_eA(_dimensionA), Jt_eB(_dimensionB);
            for (int i = 0; i < _countA; i++) makeZeroVector(_Jt_eA[i]);
            for (int j = 0; j < _countB; j++) makeZeroVector(_Jt_eB[j]);

            Matrix2x2d JpTJp; JpTJp[0][1] = 0.0; JpTJp[1][0] = 0.0;
            Vector1d Jkt_ep;

            Vector<double> Jkt_sA(_dimensionA);
            Matrix<double> paramTpA (_dimensionA, 1);

            Vector<double> Jkt_sB(_dimensionB);
            Matrix<double> paramTpB (_dimensionB, 1);

            for (int k = 0; k < nMeasurements; k++){
                int const cam = costFun._correspondingParams[k][0];
                int const point = costFun._correspondingParams[k][1];

                double const sk = computeScale(_s[k]);
                //Compute new residual res = ri/(1 + s^2);
                Vector2d res = _scaledResiduals[k];

                //Gradients for F: 
                //This is normal BA Jacobians, which has been scaled;
                multiply_At_v(Js1[k], res, Jt_eA);
                scaleVectorIP(_mu * weights[k], Jt_eA);

                multiply_At_v(Js2[k], res, Jt_eB);
                scaleVectorIP(_mu * weights[k], Jt_eB);
                
                //Compute gradients for s_i:
                Matrix2x1d Js = JacobianS(_s[k], k);
                multiply_At_v(Js, res, Jkt_ep);
                scaleVectorIP(_mu * weights[k], Jkt_ep);

                //Add the second term (the constraints of H)
                Jkt_ep[0] += _mu1 * _s[k];
                scaleVectorIP(-1.0, Jkt_ep);
                _gradS[k] = Jkt_ep[0];

                double const wk = inverseWeight(Js);
                multiply_At_B(Js1[k], Js, paramTpA);
                scaleMatrixIP(_mu * weights[k] * wk, paramTpA);
                scaleMatrixIP(_gradS[k], paramTpA);
                for (int l = 0; l < _dimensionA; l++)
                    Jkt_sA[l] = paramTpA[l][0];

                multiply_At_B(Js2[k], Js, paramTpB);
                scaleMatrixIP(_mu * weights[k] * wk, paramTpB);
                scaleMatrixIP(_gradS[k], paramTpB);
                for (int l = 0; l < _dimensionB; l++)
                    Jkt_sB[l] = paramTpB[l][0];

                addVectorsIP(Jkt_sA, Jt_eA);
                addVectorsIP(Jkt_sB, Jt_eB);


                addVectorsIP(Jt_eA, _Jt_eA[cam]);
                addVectorsIP(Jt_eB, _Jt_eB[point]);

            }

        } //end void eval Jte


        void solveSi(Vector<double> const &weights, double const alpha = 1.0){

           NLSQ_CostFunction &costFun = *_costFunctions[0];
           NLSQ_Residuals &residuals = *_residuals[0];
           Robust_NLSQ_CostFunction_Base &robustCostFun = *_robustCostFunctions[0];
           
           int const nParamTypes = costFun._usedParamTypes.size();
           int const nMeasurements = costFun._nMeasurements;
           double *delta;


           for (int k = 0; k < nMeasurements; k++){
               double gS = _gradS[k];

               Matrix2x1d Js = JacobianS(_s[k], k);
               for (int i = 0; i < nParamTypes; i++){
                   if (i == 0) delta = &_DeltaA[0]; else delta = &_DeltaB[0];
                   int const paramType = costFun._usedParamTypes[i];
                   int const paramDim = _paramDesc.dimension[paramType];
                   int const id = costFun._correspondingParams[k][i];
                   int const dstRow = id * paramDim;

                   Matrix<double> pTParam(1, paramDim);
                   MatrixArray<double> const &J = *residuals._Js[i];
                   Vector1d gpSubtraction, gP, dP;


                   Vector<double> deltaP(paramDim, delta + id * paramDim);
                   /* for (int l = 0; l < paramDim; l++) */
                   /*     deltaP[l] = delta[dstRow + l]; */

                   multiply_At_B(Js, J[k], pTParam);
                   scaleMatrixIP(weights[k] * _mu, pTParam);

                   multiply_A_v(pTParam, deltaP, gpSubtraction);
                   gS -= gpSubtraction[0];
               }

               /* double const wk = 1.0/(1 + weights[k]); */
               double const wk = inverseWeight(Js);
               double dS = wk * gS;
               /* scaleVector(wk, gP, dP); */
               _s[k] += alpha * dS;
           }
        }

        void fillHessian(Vector<double> const &weights,
                FilterStep const &filterStep = FilterStep::MOO)
        {

            Matrix<double> J1tJ1(_dimensionA, _dimensionA), 
                J2tJ2(_dimensionB, _dimensionB), J1tJ2(_dimensionA, _dimensionB);

            for (int i = 0; i < _countA; ++i) makeZeroMatrix(_Us[i]);
            for (int j = 0; j < _countB; ++j) makeZeroMatrix(_Vs[j]);
            for (int k = 0; k < _nMeasurements; ++k) makeZeroMatrix(_Ws[k]);
            NLSQ_CostFunction &costFun = *_costFunctions[0];
            NLSQ_Residuals const &residuals = *_residuals[0];

            int const nParamTypes = costFun._usedParamTypes.size();
            vector<int> const &usedParamTypes = costFun._usedParamTypes;
            Matrix2x2d JpTJp; JpTJp[0][1] = 0.0; JpTJp[1][0] = 0.0;
            for (int i1 = 0; i1 < nParamTypes; ++i1){
                int const t1 = usedParamTypes[i1], dim1 = _paramDesc.dimension[t1];
                MatrixArray<double> const &Js1 = *residuals._Js[i1];
                for (int i2=0; i2 < nParamTypes; i2++){
                    int const t2 = usedParamTypes[i2], dim2 = _paramDesc.dimension[t2];
                    MatrixArray<double> &Js2 = *residuals._Js[i2];

                    Matrix<double> J1tJ2(dim1, dim2);
                    Matrix<double> sJ1tJ2(dim1, dim2);
                    Matrix<double> J1tJs(dim1, 1);
                    Matrix<double> JstJ2(1, dim2);
                    Matrix<double> J1tP(dim1, 2);
                    /* if (!_hessian.Hs[t1][t2]) continue; */
                    MatrixArray<double> &Hs = *_hessian.Hs[t1][t2];

                    for (int k = 0; k < _nMeasurements; k++){
                        int const cam = costFun._correspondingParams[k][0];
                        int const point = costFun._correspondingParams[k][1];
                        int const ix1 = costFun._correspondingParams[k][i1];
                        int const id1 = this->getParamId(t1, ix1);
                        int const ix2 = costFun._correspondingParams[k][i2];
                        int const id2 = this->getParamId(t2, ix2);

                        if (id1 > id2) continue;
                        
                        //This is the regular JtJ
                        multiply_At_B(Js1[k], Js2[k], J1tJ2);
                        scaleMatrixIP(_mu * weights[k], J1tJ2);
                        
                        //Added terms related to si:
                        //Compute inv(JpTJp)
                        Matrix2x1d Js = JacobianS(_s[k], k);
                        multiply_At_B(Js1[k], Js, J1tJs);
                        scaleMatrixIP(_mu * weights[k], J1tJs);
                        multiply_At_B(Js, Js2[k], JstJ2);
                        scaleMatrixIP(_mu * weights[k], JstJ2);
                        multiply_A_B(J1tJs, JstJ2, sJ1tJ2);
                        double const wk = inverseWeight(Js);
                        scaleMatrixIP(wk, sJ1tJ2);


                        scaleMatrixIP(-1.0, sJ1tJ2);

                        addMatricesIP(sJ1tJ2, J1tJ2);
                        if (i1 == 0 && i2 == 0){
                            addMatricesIP(J1tJ2, _Us[cam]);
                        } else if (i1 == 1 && i2 == 1){
                            addMatricesIP(J1tJ2, _Vs[point]);
                        } else if (i1==0 && i2 == 1){
                            addMatricesIP(J1tJ2, _Ws[k]);
                        }

                    }
                }
            }//end for int i1

        }


        //Evaluating the true robust cost 
        double eval_current_cost(Vector<double> const &cached_errors)const {
            Robust_NLSQ_CostFunction_Base &robustCostFun = *_robustCostFunctions[0];
            double cost = robustCostFun.eval_target_cost(1.0, cached_errors);
            return cost;
        }

        void minimize() {

            status = LEVENBERG_OPTIMIZER_TIMEOUT;
            assert(_totalParamCount > 0);
            _filter.setAlpha(_filterAlpha);


            int const totalParamDimension = _totalParamDimension;
            vector<double> best_x(totalParamDimension);

            
            int nMeasurements = _costFunctions[0]->_nMeasurements;
            Vector<double> errors (nMeasurements);
            Vector<double> weights(nMeasurements);

            fillVector(1.0, _residuals[0]->_weights);
            this->saveAllParameters();

            Robust_NLSQ_CostFunction_Base &robustCostFun = *_robustCostFunctions[0];
            bool success_LDL, success_decrease, accepted;
            double f, h, f1, h1;

            
            robustCostFun.cache_residuals(*_residuals[0], errors);
            for (int k = 0; k < nMeasurements; k++){
                _s[k] = _initialScale;
            }

            robustCostFun.cache_residuals(*_residuals[0], errors);
            this->evalFH(f, h);
            _filter.addElement(f, h);
            FilterStep filterStep = FilterStep::MOO;
            _timer.reset(); _timer.start();

            int iter = 0;
            /* for (int iter = 0; iter < maxIterations; iter++){ */
            while (iter < maxIterations) {
                iter++;
                //Evaluate the residuals 
                robustCostFun.cache_residuals(*_residuals[0], errors);
                this->evalFH(f, h);

                double const initFH = f + h;
                double const initial_cost = this->eval_current_cost(errors);
                this->updateStat(initial_cost);

                //Cache weights for pi
                robustCostFun.cache_weight_fun(1.0, _scaledErrors, weights);

                std::cout << "Filter iteration : " << iter << " robust cost = "  << initial_cost
                    << "f : " << f << " h: " << h << " initial FH : " << initFH <<
                    " damping = " << damping_value << " step : " << filterStep << std::endl;

                
                this->fillJacobians();
                scaleJacobians();

                if (filterStep == FilterStep::MOO){
                    _mu = _smu; 
                    _mu1 = 1 - _mu;
                } else {
                   _mu = 0.0; _mu1 = 1.0;
                }

                this->evalJt_e(weights, filterStep);
                this->fillHessian(weights, filterStep);
                this->addDamping();
                this->prepare_schur();
                this->fillJt_e_schur();
                this->fill_M();
                this->solveJtJ();

                success_LDL = true;
                success_decrease = false;
                
                /* if (isnan(_deltaA[0][0])) */
                /*     success_LDL = false; */

                if (success_LDL)
                {
                    this->saveAllParameters();
                    storeS();

                    //update P variables
                    double alpha = 1.0;
                    if (filterStep == FilterStep::EXPLORATION)
                        alpha = _explorationStepSize;
                    solveSi(weights, alpha);

                    if (filterStep == FilterStep::MOO){
                        this->updateParameters(0, _deltaA);
                        this->updateParameters(1, _deltaB);
                        this->finishUpdateParameters();
                    }

                    robustCostFun.cache_residuals(*_residuals[0], errors);
                    double const new_cost = robustCostFun.eval_target_cost(1.0, errors);
                    this->evalFH(f1, h1);

                    double const newFH = f1 + h1;
                    accepted = _filter.isAccepted(f1, h1);
                    success_decrease = newFH < initFH;

                } else 
                {
                    damping_value = damping_value * 10;
                    continue;
                }

                if (filterStep == FilterStep::MOO){
                    if (accepted){
                        damping_value = std::max(1e-8, damping_value * 0.1);
                        _filter.addElement(f1, h1);
                    } else {
                        cout << " Restoration ... \n";
                        iter--;
                        damping_value *= 10;
                        if (success_LDL) {
                            this->restoreAllParameters();
                            restoreS();
                        }

                        //Conduct Exploration here 
                        /* filterStep = FilterStep::EXPLORATION; */
                        Vector<double> gradH(_nMeasurements);
                        //Perform grid search 
                        double minAngle = 1e20, bestGamma;
                        double gamma = -0.3;
                        storeS();
                        while (gamma >= -0.5){
                            for (int k = 0; k < _nMeasurements; k++){
                                _s[k] += gamma *_s[k]; 
                                gradH [k] = _s[k];
                            }
                            this->evalFH(f, h);
                            /* scaleVectorIP(-1.0,_gradS); */
                            //Cache weights for pi
                            robustCostFun.cache_weight_fun(1.0, _scaledErrors, weights);
                            this->evalJt_e(weights, filterStep);
                            double  angle = innerProduct(gradH, _gradS)
                                /((norm_L2(gradH) * norm_L2(_gradS)));
                            angle = acos(angle);

                            /* std::cout <<std::setw(5)<< " gamma = " << gamma << " New h = " << h */  
                            /*     << " angle = " << angle << std::endl; */
                            if (angle < minAngle ){
                                minAngle = angle; bestGamma = gamma;
                            }
                            gamma -= 0.1;
                            restoreS();
                        }

                        for (int k = 0; k < _nMeasurements; k++){
                            _s[k] += bestGamma *_s[k]; 
                        }
                    }
                } 
                /* else //Filter Step = EXPLORATION */
                /* { */
                /*     if (success_decrease){ */
                /*         damping_value = std::max(1e-8, damping_value * 0.1); */
                /*         if (accepted){ */
                /*             filterStep = FilterStep::MOO; */
                /*         } */
                /*     } else { */
                /*         damping_value *= 10; */
                /*         if (success_LDL) { */
                /*             this->restoreAllParameters(); */
                /*             restoreS(); */
                /*         } */
                /*     } */
                /* } */
            } //end for iter
        } //end void minimize


        Filter _filter;
        double &damping_value = _damping;
        double damping_constraints = 1000.0;
        /* double damping_constraints = 100.0; */
        double _smu = 0.9, _smu1 = 1 -_smu;
        double _initialScale =  50.0;
        /* double _initialScale =  50.0; */
        double _explorationStepSize = 0.0001;

        protected:
            double _mu, _mu1;
            /* double _filterAlpha = 0.01; */
            double _filterAlpha = 0.01;
            Vector<double> _s, _storeS;
            Vector<double> _scaledErrors;
            Vector<double> _gradS;
            std::vector<Vector2d> _scaledResiduals;
            int _nMeasurements;
            /* std::ofstream &_log_file; */

   };



};//end namespace 

#endif

