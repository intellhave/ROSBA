#ifndef ROBUST_LSQ_SCHUR_PCG_FILTER_ASKER_GEMM_H
#define ROBUST_LSQ_SCHUR_PCG_FILTER_ASKER_GEMM_H

#include "robust_lsq_common.h"
#include "Math/v3d_linearbase.h"
#include "Math/v3d_nonlinlsq.h"
#include "utilities_common.h"
#include "Math/v3d_linear_lu.h"
#include "Math/v3d_linear_ldlt.h"

#include "robust_lsq_schur_pcg_filter_asker.h"
#include "robust_lsq_lifted.h"
#include <iomanip>
#include <fstream>
#include <random>

#include "filter.h"

namespace Robust_LSQ{

    struct Robust_LSQ_Optimizer_Schur_PCG_Filter_GeMM : public Robust_LSQ_Optimizer_Schur_PCG_Filter_Asker {
        typedef Robust_LSQ_Optimizer_Schur_PCG_Filter_Asker Base;
        Robust_LSQ_Optimizer_Schur_PCG_Filter_GeMM(NLSQ_ParamDesc const &paramDesc,     
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions): 
            Base(paramDesc, costFunctions, robustCostFunctions){


            }

        typedef Lifted_Smooth_Truncated Lifted_Kernel;
        //Implement a new minimization procedure that uses GeMM for exploration

        void minimize() {

            status = LEVENBERG_OPTIMIZER_TIMEOUT;
            assert(_totalParamCount > 0);
            _filter.setAlpha(_filterAlpha);

            Lifted_Kernel lift_kernel(1.0);

            int const totalParamDimension = _totalParamDimension;
            vector<double> best_x(totalParamDimension);

            
            int nMeasurements = _costFunctions[0]->_nMeasurements;
            Vector<double> errors (nMeasurements);
            Vector<double> weights(nMeasurements);
            Vector<double> irlsWeights(nMeasurements);

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

            for (int iter = 0; iter < maxIterations; iter++){
                //Evaluate the residuals 
                robustCostFun.cache_residuals(*_residuals[0], errors);
                this->evalFH(f, h);


                double const initFH = f + h;
                double const initial_cost = this->eval_current_cost(errors);

                //Cache weights for pi
                robustCostFun.cache_weight_fun(1.0, _scaledErrors, weights);
                this->updateStat(initial_cost);

                std::cout << "iteration : " << iter << " robust cost = "  << initial_cost
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
                //Update parameters
                this->saveAllParameters();
                this->storeS();

                success_LDL = true;
                success_decrease = false;

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

                    this->finishUpdateParameters();
                    robustCostFun.cache_residuals(*_residuals[0], errors);
                    double const new_cost = robustCostFun.eval_target_cost(1.0, errors);
                    this->evalFH(f1, h1);
                    double const newFH = f1 + h1;
                    accepted = _filter.isAccepted(f1, h1);
                    success_decrease = newFH < initFH;
                } 

                if (filterStep == FilterStep::MOO){
                    if (accepted){
                        damping_value = std::max(1e-8, damping_value * 0.1);
                        _filter.addElement(f1, h1);

                    } else { //not accepted -> Using GeMM to determine new scales
                        this->restoreAllParameters();
                        this->restoreS();
                        robustCostFun.cache_residuals(*_residuals[0], errors);
                        robustCostFun.cache_weight_fun(1.0, errors, irlsWeights);
                        this->evalFH(f, h);
                        double const robust_cost = robustCostFun.eval_target_cost(1.0, errors);

                        //Cache weights of scaled residuals
                        robustCostFun.cache_weight_fun(1.0, _scaledErrors, weights);
                        //compute the upper bound 
                        double lifted_cost = 0;
                        double sigma_start = 1, sigma_end = 10;
                        for (int k = 0; k < _nMeasurements; k++){
                            double sk = (1.0 + sqr(_s[k]));
                            Vector<double> er(1), wk(1);
                            er[0] = errors[k];
                            robustCostFun.cache_weight_fun(sk, er, wk);
                            lifted_cost += 0.5 * wk[0] * errors[k] + lift_kernel.gamma(wk[0]);
                        }

                        double const target_cost = gemm_mu * robust_cost + (1 - gemm_mu) * lifted_cost;
                        //Start bisection
                        double new_cost = 0;
                        double middle_cost = 0.5* robust_cost + 0.5*target_cost;
                        this->storeS();
                        int count = 0;
                        double sigma = 1.0;
                        while (new_cost <= middle_cost || new_cost > target_cost)
                        {
                            sigma= 0.5*(sigma_start + sigma_end);
                            /* sigma += 0.02; */

                            this->restoreS();
                            for (int k = 0; k < _nMeasurements; k++){
                                _s[k] = _s[k]/sigma;
                            }
                            /* this->evalFH(f, h); */
                            /* _robustCostFunctions[0]->cache_weight_fun(1.0, */ 
                            /*         _scaledErrors, weights); */

                            new_cost = 0;
                            for (int k = 0; k < _nMeasurements; ++k)
                            {
                                double sk = (1.0 + sqr(_s[k]));
                                Vector<double> er(1), wk(1);
                                er[0] = errors[k];
                                robustCostFun.cache_weight_fun(sk, er, wk);
                                new_cost += 0.5 * wk[0] * errors[k] + lift_kernel.gamma(wk[0]);
                            }
                            count++;
                            std::cout << sigma << " " << new_cost << std::endl;
                            if (count % 100000 == 0)
                                cout << "iter  = " << currentIteration << " sigma_start = " << sigma_start << " sigma end = " << sigma_end 
                                    << " sigma = " << sigma << " expected cost = " <<  target_cost << " cost = " << new_cost << endl;
                            if (middle_cost <= new_cost && new_cost <= target_cost)
                                break;

                            if (new_cost < middle_cost)//scale too large
                                sigma_start = sigma;
                            else // new_cost > target_cost: scale too small
                                sigma_end = sigma;

                        } //end while 

                    }
                } 
                else //Filter Step = EXPLORATION
                {


                }

            } //end for iter
        } //end void minimize


        double const gemm_mu = 0.5;





    };//end struct definition



};//end namespace

#endif
