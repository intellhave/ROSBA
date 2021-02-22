#ifndef ROBUST_LSQ_SCHUR_PCG_GNC_H
#define ROBUST_LSQ_SCHUR_PCG_GNC_H

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

    template <int NLevels, bool allow_early_level_stopping = true>
    struct Robust_LSQ_Optimizer_Schur_PCG_GNC: public Robust_LSQ_Optimizer_Schur_PCG_Base{

        typedef Robust_LSQ_Optimizer_Schur_PCG_Base Base;

        static int  constexpr GNC_mode = 0;
        static bool constexpr force_level_stopping = 0;

        static double constexpr alpha_multiplier = 2;
        static double constexpr eta_multiplier = 2;
        Robust_LSQ_Optimizer_Schur_PCG_GNC(NLSQ_ParamDesc const &paramDesc,     
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions): 
            Base(paramDesc, costFunctions, robustCostFunctions),
            _cvx_xs(robustCostFunctions.size()), _alphas(robustCostFunctions.size())
        {
            this->fill_alphas_etas();
        }

        double _etas[NLevels], _etas2[NLevels];
        vector<double> _cvx_xs;
        vector<InlineVector<double, NLevels> > _alphas;

         virtual void fill_alphas_etas()
         {
            for (int obj = 0; obj < _robustCostFunctions.size(); ++obj)
            {
               double const cvx_x = _robustCostFunctions[obj]->get_convex_range(1.0);
               _cvx_xs[obj] = cvx_x;

               auto &alphas = _alphas[obj];

               alphas[0] = 1.0;
               for (int k = 1; k < NLevels; ++k) alphas[k] = alpha_multiplier * alphas[k-1];
               for (int k = 0; k < NLevels; ++k) alphas[k] -= 1.0;
               for (int k = 0; k < NLevels; ++k) alphas[k] *= cvx_x;
            } // end for (obj)

            _etas[0] = 1.0;
            for (int k = 1; k < NLevels; ++k) _etas[k]  = eta_multiplier * _etas[k-1];
            for (int k = 0; k < NLevels; ++k) _etas2[k] = sqr(_etas[k]);
         }//end fill_alpha_etas

         double eval_current_cost(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            return cost;
         } //end eval_current_cost


         void cache_target_costs(int const obj, int const level, Vector<double> const& errors, Vector<double> &costs) const
         {
             _robustCostFunctions[obj]->cache_target_costs(_etas2[level], errors, costs);
         } //end cache_target_cost 


         double eval_level_cost(int const level, vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;

            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto costFun = _robustCostFunctions[obj];
               auto const& errors = cached_errors[obj];

               int const K = errors.size();
               Vector<double> costs(K);
               this->cache_target_costs(obj, level, errors, costs);
               double cost1 = 0; 
               for (int k = 0; k < K; ++k) 
                  cost1 += costs[k];
               cost += cost1;
            } // end for (obj)
            return cost;
         } //end eval_level_cost 


         void compute_weights(int obj, int level, Vector<double> const& errors, Vector<double>& weights) const
         {
            auto costFun = _robustCostFunctions[obj];
            int const K = errors.size();

            double const s2 = _etas2[level];
            for (int k = 0; k < K; ++k)
            {
                double const r2 = errors[k];
                weights[k] = costFun->eval_target_weight(s2, r2);
            } // end for (k)
         }//end compute_weights

         double eval_robust_objective() const
         {
            int const nObjs = _robustCostFunctions.size();
            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            return this->eval_level_cost(0, cached_errors);
         } //end eval_robust_objective;

         virtual bool allowStoppingCriteria() const { return true; }


         Vector2d eval_level_deltas(int const level, vector<Vector<double> > const& cached_errors, vector<Vector<double> > const& cached_errors0) const
         {
            int const nObjs = _robustCostFunctions.size();

            Vector2d res(0.0, 0.0);

            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto const& errors  = cached_errors[obj];
               auto const& errors0 = cached_errors0[obj];

               int const K = errors.size();

               Vector<double> costs_old(K), costs_new(K);
               this->cache_target_costs(obj, level, errors0, costs_old);
               this->cache_target_costs(obj, level, errors,  costs_new);

               for (int k = 0; k < K; ++k)
               {
                  res[0] += std::max(0.0, costs_old[k] - costs_new[k]);
                  res[1] += std::max(0.0, costs_new[k] - costs_old[k]);
               } // end for (k)
            } // end for (obj)

            return res;
         } // end eval_level_deltas()


         void minimize() 
         {

            double const rho_stopping = 0.95;
            status = LEVENBERG_OPTIMIZER_TIMEOUT;
            int const totalParamDimension = _totalParamDimension;
            int const nObjs = _robustCostFunctions.size();
            double &damping_value = this->_damping;

            vector<Vector<double> > cached_errors(nObjs), cached_errors0(nObjs);
            vector<Vector<double> > cached_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) {
                cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_errors0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
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

            currentIteration = 0;
            for (int level = NLevels-1; level >= 0; --level)
            {

               int const remainingIterations = maxIterations - currentIteration;
               int const n_inner_iterations = force_level_stopping ? int(0.5+double(remainingIterations)/(level+1)) : remainingIterations;

               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {

                  for (int obj = 0; obj < nObjs; ++obj) 
                     _robustCostFunctions[obj]->cache_residuals(
                             *_residuals[obj], cached_errors[obj]);

                  double const Psi0_k  = this->eval_level_cost(level, cached_errors);
                  double const Psi0_kk = (level > 0) ? this->eval_level_cost(level-1, cached_errors) : 0.0;

                  for (int obj = 0; obj < nObjs; ++obj) 
                      cached_errors0[obj] = cached_errors[obj];

                  //Compute cost to log
                  double const init_cost = this->eval_level_cost(0, cached_errors);
                  this->updateStat(init_cost);


                  this->fillJacobians();

                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     this->compute_weights(obj, level, cached_errors[obj], 
                             cached_weights[obj]);

                  }

                  this->evalJt_e(cached_weights);
                  this->fillHessian(cached_weights);
                  this->addDamping();
                  this->prepare_schur();
                  this->fillJt_e_schur();
                  this->fill_M();
                  this->solveJtJ();
                  ++currentIteration;
                  this->saveAllParameters();

                  this->updateParameters(0, _deltaA);
                  this->updateParameters(1, _deltaB);
                  this->finishUpdateParameters();

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
                  double const Psi1_k  = this->eval_level_cost(level, cached_errors);

                  bool success_decrease = (Psi1_k < Psi0_k);
                  bool stopping_reached = false;

                  if (allow_early_level_stopping && level > 0)
                  {
                      double const Psi1_kk = (level > 0) ? this->eval_level_cost(level-1, cached_errors) : 0.0;
#if 0
                      double const rho = (Psi0_kk - Psi1_kk) / (Psi0_k - Psi1_k);
                      if (rho > rho_stopping) stopping_reached = true;
#elif 0
                      double const eta = 0.5, th_lo = 0.5*(eta-1.0)/eta, th_hi = 0.5*(eta+1.0)/eta;
                      double const rho = (Psi0_kk - Psi1_kk) / (Psi0_k - Psi1_k);
                      //cout << "rho = " << rho << " th_lo = " << th_lo << " th_hi = " << th_hi << endl;
                      if (rho < th_lo || rho > th_hi) stopping_reached = true;
#elif 1
                      double const eta = 0.1; // 0.2
                      Vector2d const deltas = this->eval_level_deltas(level, cached_errors, cached_errors0);
                      double const rho = (deltas[0] - deltas[1]) / (deltas[0] + deltas[1]);
                      if (rho < eta) stopping_reached = true;
#else
                      double const rho = fabs(Psi0_kk - Psi1_kk) - (Psi0_k - Psi1_k);
                      if (rho > 0.0) stopping_reached = true;
#endif
                  } // end if

                  if (optimizerVerbosenessLevel >= 1)
                  {
                      double const current_cost = this->eval_level_cost(0, cached_errors);
                      // if (optimizerVerbosenessLevel == 1 && (1 || success_decrease))
                      //    cout << "Fast_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost << " lambda = " << damping_value << endl;
                      if (optimizerVerbosenessLevel >= 1 && (0 || success_decrease))
                      {
                          cout << "Fast_HOM_Optimizer: iteration " << setw(3) << currentIteration << " level = " << level << " new cost = " << setw(12) << current_cost
                             << " Psi_k(prev) = " << setw(12) << Psi0_k << " Psi_k(new) = " << setw(12) << Psi1_k
                              << " success_decreas = " << int(success_decrease) << " lambda = " << damping_value << endl;


                      }
                  } //end if optimize verbosenessLEvel

                  if (success_decrease)
                  {
                     damping_value = std::max(damping_value_min, damping_value / 10);
                  }
                  else
                  {
                     this->restoreAllParameters();
                     damping_value *= 10;
                  }

                  if (level > 0 && stopping_reached) break;

               }//end for iter

            } // for nlevel

         }//end void minimize()


    };//end struct definition



};//end namespace

#endif

