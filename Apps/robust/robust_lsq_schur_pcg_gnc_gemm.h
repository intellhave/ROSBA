#ifndef ROBUST_LSQ_SCHUR_PCG_GNC_GEMM_H
#define ROBUST_LSQ_SCHUR_PCG_GNC_GEMM_H

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

namespace Robust_LSQ{
    
    struct Robust_LSQ_Optimizer_Schur_PCG_GNC_GeMM: public Robust_LSQ_Optimizer_Schur_PCG_Base{

        typedef Robust_LSQ_Optimizer_Schur_PCG_Base Base;

        static int  constexpr GNC_mode = 0;
        static bool constexpr force_level_stopping = 1;

        static double constexpr alpha_multiplier = 2;
        static double constexpr eta_multiplier = 2;
        Robust_LSQ_Optimizer_Schur_PCG_GNC_GeMM(NLSQ_ParamDesc const &paramDesc,     
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions): 
            Base(paramDesc, costFunctions, robustCostFunctions),
            _s(costFunctions[0]->_nMeasurements), _storeS(costFunctions[0]->_nMeasurements)
        {
            //initialize scales
            for (int i = 0; i < _nMeasurements; i++){
                _s[i] = _initScale;
            }
        }


        void storeS(){ for (int k = 0; k < _nMeasurements; k++){ _storeS[k] = _s[k];}}
        void restoreS(){ for (int k = 0; k < _nMeasurements; k++){ _s[k] = _storeS[k];}}
        double eval_current_cost(vector<Vector<double> > const& cached_errors) const
        {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) 
                cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            return cost;
        } //end eval_current_cost


        double evalScale2(double const &s) const {
            return 1 + s*s;
        }

        double evalRobustCost(double const &error){
            return _robustCostFunctions[0]->eval_target_fun(1.0, error);
        }

        double scaledError(double const &error, double const &scale2){
            Vector<double> err(1); err[0] = error;
            return _robustCostFunctions[0]->eval_target_fun(scale2, error);
        }

        double scaledWeight(double const &error, double const &scale2){
            return _robustCostFunctions[0]->eval_target_weight(scale2, error);
        }
        
        double evalUpperBound(double const &error, double const &scale2){
            return scale2 * scaledWeight(error, scale2);
        }

        double evalAllScaledCost(vector<Vector<double>> const &cached_errors){
            double cost = 0.0;
            for (int k = 0; k < _nMeasurements; k++){
                double const err_k = cached_errors[0][k];
                double const newScale2 = evalScale2(_s[k]);
                /* cost += newScale2 * scaledError(err_k, newScale2); */
                cost += scaledError(err_k, newScale2);
            }
            return cost;
        }


        void minimize(){

            int const totalParamDimension = _totalParamDimension;
            int const nObjs = _robustCostFunctions.size();
            double &damping_value = this->_damping;

            vector<Vector<double> > cached_errors(nObjs);
            vector<Vector<double> > cached_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) {
                cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                fillVector(1.0, _residuals[obj]->_weights);
            }

            for (int obj = 0; obj < nObjs; ++obj) 
                _robustCostFunctions[obj]->cache_residuals(
                        *_residuals[obj], cached_errors[obj]);

            double startingCost = this->eval_current_cost(cached_errors);
            cout << "Initialization cost  = " << startingCost << endl;
            _timer.reset(); _timer.start();
            this->updateStat(startingCost);
            this->saveAllParameters();

            //S
            double Jtu[100000];
            int iter = 0;
            while (iter < maxIterations){
            /* for (int iter = 0; iter < maxIterations; iter++){ */
                iter++;

                _robustCostFunctions[0]->cache_residuals(
                        *_residuals[0], cached_errors[0]);

                double initial_cost = this->eval_current_cost(cached_errors);
                std::cout << "iter : " << iter << " robust cost = " << initial_cost << 
                    " damping = " << damping_value << endl;
                this->updateStat(initial_cost);
                this->storeS();
                

                //Update the scales; 
                double initScaledCost = 0;
                for (int k = 0; k < _nMeasurements; k++){

                    //Compute real cost vs ``lifted" cost 
                    double const err_k = cached_errors[0][k];
                    double const robustCost = evalUpperBound(cached_errors[0][k], 0);
                    double const scale2 = evalScale2(_s[k]);
                    double const bound = evalUpperBound(cached_errors[0][k], scale2);
                    double const target_cost = _mu * robustCost + (1-_mu) * bound;
                    double const middle_cost = 0.5 * robustCost + 0.5 * target_cost;

                    double curr_cost = 0;
                    double sk = _s[k];
                    double sigmaStart = 0.0, sigmaEnd = sk;
                    /* cout << iter << " " << k << std::endl; */
                    while (true){
                        /* double sigma = 0.5 * (sigmaStart + sigmaEnd); */
                        double sigma = 0.001;
                        _s[k] = _s[k] - sigma * _s[k];
                        double currScale = evalScale2(_s[k]);
                        double const currCost = evalUpperBound(err_k, currScale);
                        /* cout << " iter : "<< iter << " k: " << k << " " << sigma << " " << currCost << " " */ 
                        /*     << middle_cost << " " << target_cost << std::endl; */
                        /* if (currCost >= middle_cost && currCost <=target_cost){ */
                        if (currCost <=target_cost || _s[k] <= 1e-3){
                            break;
                        }

                        if (abs(currCost - middle_cost) < 1e-6 || abs(currCost -target_cost) < 1e-6){
                            break;
                        }

                        /* if (currCost > target_cost){ */
                        /*     sigmaEnd = sigma; */
                        /* } else if (currCost < middle_cost){ */
                        /*     sigmaStart = sigma; */
                        /* } */
                    }

                    //Cache Weights
                    double const newScale2 = evalScale2(_s[k]);
                    /* cached_weights[0][k] = newScale2 * scaledWeight(err_k, newScale2); */
                    cached_weights[0][k] = scaledWeight(err_k, newScale2);

                } //end for k

                initScaledCost = evalAllScaledCost(cached_errors);;

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

                _robustCostFunctions[0]->cache_residuals(
                        *_residuals[0], cached_errors[0]);
                double newScaledCost = evalAllScaledCost(cached_errors);

                if (newScaledCost <= initScaledCost){
                    damping_value = max(1e-8, damping_value * 0.1);
                } else {
                    damping_value *= 10;
                    this->restoreAllParameters();
                    this->restoreS();
                    iter--;
                }

            }

        }

        double _mu = 0.9;
        double _initScale = 50;
        protected:
        Vector<double> _s, _storeS;

    };//end struct definition 



};//end namespace 



#endif
