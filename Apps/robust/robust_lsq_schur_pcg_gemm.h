#ifndef ROBUST_LSQ_SCHUR_PCG_GEMM_H
#define ROBUST_LSQ_SCHUR_PCG_GEMM_H

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


namespace Robust_LSQ{

    struct Robust_LSQ_Optimizer_Schur_PCG_GeMM: public Robust_LSQ_Optimizer_Schur_PCG_Base{

        typedef Robust_LSQ_Optimizer_Schur_PCG_Base Base;
        typedef Lifted_Smooth_Truncated Lifted_Kernel;

        Robust_LSQ_Optimizer_Schur_PCG_GeMM(NLSQ_ParamDesc const &paramDesc,     
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions): 
            Base(paramDesc, costFunctions, robustCostFunctions)
        {

        }//end constructor


        void minimize(){

            int const nObjs = _robustCostFunctions.size();
            double &damping_value = _damping;
            vector<Vector<double>> cached_errors(nObjs);
            vector<Vector<double>> cached_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj){
                cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            }

            //Initialize all weights to 1.0
            for (int obj = 0; obj < nObjs; ++obj){
                fillVector(1.0, _residuals[obj]->_weights);
                fillVector(1.0, cached_weights[obj]);
            }

            Lifted_Kernel lift_kernel(1.0);
            
            //Vector to store J(theta, u) over iterations
            double Jtu[100000];
            _timer.start();
            double lifted_cost = -1.0;
            for (currentIteration = 0; currentIteration < maxIterations; currentIteration++)
            {

                //Compute the cost at the beginning of each iteration
                for (int obj = 0; obj < nObjs; ++obj){
                    
                    Robust_NLSQ_CostFunction_Base const &robustCostFun = *_robustCostFunctions[obj];
                    robustCostFun.cache_residuals(*_residuals[obj], cached_errors[obj]);
                    /* robustCostFun.cache_weight_fun(1.0, cached_errors[obj], cached_weights[obj]); */
                }
                double const initial_cost = this->eval_current_cost(cached_errors);


                if (optimizerVerbosenessLevel >= 1)
                    cout << "GeMM Optimizer: iteration: " << setw(3) << currentIteration
                        << ", initial |residual|^2 = " << setw(12) << initial_cost
                        << " lambda = " << damping_value << std::endl;

                //Compute IRLS Weights
                bool success_LDL = true;
                double const robust_cost = initial_cost;
                //Note: Right now we assume that nObjs =  1
                for (int obj = 0; obj < nObjs; ++obj)
                {
                    NLSQ_Residuals &residuals = *_residuals[obj];
                    int const K = _nMeasurements;

                    //Modify the weights
                    double sigma_start = 1.0, sigma_end = 1e8;

                    //Very first iteration
                    if (lifted_cost < 0){
                        for (int k = 0; k < K; ++k)
                        {
                            double const w = cached_weights[obj][k];
                            lifted_cost += 0.5 * w * cached_errors[obj][k] 
                                + lift_kernel.gamma(w);
                        }
                    } 


                    double const total_cost = mu * robust_cost +  (1 - mu) * lifted_cost;

                    double v = total_cost;
                    cout << " Robust cost = " << robust_cost  
                        << " lifted_cost = " << lifted_cost 
                        << " total cost =  " << total_cost << endl;

                    this->updateStat(robust_cost);


                    //Perform bisection to find the right value of sigma
                    double new_cost = 0;
                    int count = 0;
                    double middle_cost = 0.5* robust_cost + 0.5*v;
                    // while (abs(new_cost - v ) > 1e-8)
                    while (new_cost <= middle_cost || new_cost > v)
                    {
                        new_cost = 0;
                        double sigma= 0.5*(sigma_start + sigma_end);
                        _robustCostFunctions[obj]->cache_weight_fun(sigma, cached_errors[obj], cached_weights[obj]);
                        for (int k = 0; k < K; ++k)
                        {
                            double const w = cached_weights[obj][k];
                            new_cost += 0.5 * w * cached_errors[obj][k] + lift_kernel.gamma(w);
                        }
                        count++;
                        if (count % 10000 == 0)
                            cout << "iter  = " << currentIteration << " sigma_start = " << sigma_start << " sigma end = " << sigma_end 
                                << " sigma = " << sigma << " expected cost = " <<  v << " cost = " << new_cost << endl;
                        if (new_cost > v)
                            sigma_end = sigma;
                        else 
                            sigma_start = sigma;

                        if (new_cost >=middle_cost && new_cost <=v){
                            lifted_cost= new_cost;
                            break;
                        }
                        if (abs(sigma-1e8)<1e-6){
                            _robustCostFunctions[obj]->cache_weight_fun(10, cached_errors[obj], cached_weights[obj]);
                            break;
                        }

/*                         if (middle_cost < new_cost && new_cost < v) */
/*                             break; */
                    } //end while 
                } //end for obj

                this->fillJacobians();
                this->evalJt_e(cached_weights);
                this->fillHessian(cached_weights);
                this->addDamping();
                this->prepare_schur();
                this->fillJt_e_schur();
                this->fill_M();
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


                if (new_err < initial_cost){
                    _damping = max(1e-8, _damping * 0.1);
                } else {
                    this->restoreAllParameters();
                    _damping = _damping * 10;
                }


            }//for iteration



        } //end minimize

        double const mu = 0.5;



    };//end struct definition
} //end namespace 


#endif

