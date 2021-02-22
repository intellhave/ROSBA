#ifndef ROBUST_LSQ_SCHUR_PCG_GNC_MOO_H
#define ROBUST_LSQ_SCHUR_PCG_GNC_MOO_H

#include "robust_lsq_common.h"
#include "Math/v3d_linearbase.h"
#include "Math/v3d_nonlinlsq.h"
#include "utilities_common.h"
#include "Math/v3d_linear_lu.h"
#include "Math/v3d_linear_ldlt.h"

/* #include "robust_lsq_schur_pcg_common.h" */
#include "robust_lsq_schur_pcg_gnc.h"
#include "robust_lsq_lifted.h"
#include <iomanip>
#include <fstream>
#include <random>


namespace Robust_LSQ{


    template <int NLevels>
    struct Robust_LSQ_Optimizer_Schur_PCG_GNC_MOO: public Robust_LSQ_Optimizer_Schur_PCG_GNC<NLevels>{

        typedef Robust_LSQ_Optimizer_Schur_PCG_GNC<NLevels> Base;

        using Base::_robustCostFunctions;
        using Base::_deltaA;
        using Base::_deltaB;
        using Base::_residuals;
        using Base::_totalParamDimension;
        using Base::_dimensionA;
        using Base::_dimensionB;
        using Base::_countA;
        using Base::_countB;
        using Base::_Jt_eA;
        using Base::_Jt_eB;
        using Base::currentIteration;

        static double constexpr beta = 0.5, beta1 = 1.0 - beta;

        static bool constexpr only_distort_gradient = 0;
        static bool constexpr use_random_distortion = 0;
        static int  constexpr GNC_mode = 0;
        static bool constexpr force_level_stopping = 1;

        double const cosine_threshold = -0.95; //-0.95;
        double const g_ratio_threshold = 0.1;
        bool const precondition_g1 = 1;

        static double constexpr alpha_multiplier = 2;
        static double constexpr eta_multiplier = 2;
        Robust_LSQ_Optimizer_Schur_PCG_GNC_MOO(NLSQ_ParamDesc const &paramDesc,     
                std::vector<NLSQ_CostFunction *> const &costFunctions,
                std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions): 
            Base(paramDesc, costFunctions, robustCostFunctions)
        {
            this->fill_alphas_etas();
        }


         virtual bool allowStoppingCriteria() const { return true; }

         static double get_scale1(double const beta, 
                 double const sqrNorm_g0, double const sqrNorm_g1, double const dot_g0_g1)
         {
            if (sqrNorm_g1 < 1e-12) return 0.0;
            if (beta == 0.5) return sqrt(sqrNorm_g0/sqrNorm_g1);
            double const a = beta*sqrNorm_g1, b = (1.0-2.0*beta)*dot_g0_g1, 
                   c = (beta-1.0)*sqrNorm_g0;
            double const D = sqrt(std::max(0.0, b*b - 4.0*a*c));
            return 0.5*(D - b)/a;
         }

         void minimize() 
         {

            double const rho_stopping = 0.95;
            this->status = LEVENBERG_OPTIMIZER_TIMEOUT;
            int const totalParamDimension = _totalParamDimension;
            int const nObjs = _robustCostFunctions.size();
            double &damping_value = this->_damping;

            vector<Vector<double> > cached_errors(nObjs), cached_errors0(nObjs);
            vector<Vector<double> > cached_weights0(nObjs), cached_weights1(nObjs);
            vector<Vector<double> > cached_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) {
                cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_errors0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_weights0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                cached_weights1[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
                fillVector(1.0, _residuals[obj]->_weights);
            }

            for (int obj = 0; obj < nObjs; ++obj) 
                _robustCostFunctions[obj]->cache_residuals(
                        *_residuals[obj], cached_errors[obj]);

            double startingCost = this->eval_current_cost(cached_errors);
            cout << "Initialization cost  = " << startingCost<< endl;


            this->saveAllParameters();
            this->currentIteration = 0;
            VectorArray<double> Jt_eA0(_countA, _dimensionA);
            VectorArray<double> Jt_eA1(_countA, _dimensionA);
            VectorArray<double> Jt_eB0(_countB, _dimensionB);
            VectorArray<double> Jt_eB1(_countB, _dimensionB);
            this->_timer.reset(); this->_timer.start();
            this->updateStat(startingCost);

            for (int level = NLevels-1; level >= 0; --level)
            {

               int const remainingIterations = this->maxIterations - this->currentIteration;
               int const n_inner_iterations = force_level_stopping ? int(0.5+double(remainingIterations)/(level+1)) : remainingIterations;

               int const level0 = 0;
               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {

                  for (int obj = 0; obj < nObjs; ++obj) 
                     _robustCostFunctions[obj]->cache_residuals(
                             *_residuals[obj], cached_errors[obj]);

                  double const Psi0_k  = this->eval_level_cost(level, cached_errors);

                  double const Psi0_k0 = (level > 0) ? 
                      this->eval_level_cost(level0, cached_errors) : Psi0_k;


                  for (int obj = 0; obj < nObjs; ++obj) 
                      cached_errors0[obj] = cached_errors[obj];

                  //Compute cost to log
                  double const init_cost = this->eval_level_cost(0, cached_errors);
                  this->updateStat(init_cost);


                  this->fillJacobians();

                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     this->compute_weights(obj, level0, cached_errors[obj], 
                             cached_weights0[obj]);
                     this->compute_weights(obj, level, cached_errors[obj], 
                             cached_weights1[obj]);
                  }

                  double scale1 = 0;
                  if (level > 0){
                      double sqrNorm_g0 = 0, sqrNorm_g1 = 0;
                      double gInnerProd = 0;
                      this->evalJt_e(cached_weights0);
                      copyVectorArray(_Jt_eA, Jt_eA0);
                      copyVectorArray(_Jt_eB, Jt_eB0);

                      this->evalJt_e(cached_weights1);
                      copyVectorArray(_Jt_eA, Jt_eA1);
                      copyVectorArray(_Jt_eB, Jt_eB1);

                      for (int i = 0; i < _countA; i++){
                          sqrNorm_g0 += sqrNorm_L2(Jt_eA0[i]);
                          sqrNorm_g1 += sqrNorm_L2(Jt_eA1[i]);
                          gInnerProd += innerProduct(Jt_eA0[i], Jt_eA1[i]);
                      }
                      for (int i = 0; i < _countB; i++){
                          sqrNorm_g0 += sqrNorm_L2(Jt_eB0[i]);
                          sqrNorm_g1 += sqrNorm_L2(Jt_eB1[i]);
                          gInnerProd += innerProduct(Jt_eB0[i], Jt_eB1[i]);
                      }
                      double const cos_01 = gInnerProd / std::max(1e-8, sqrt(sqrNorm_g0 * sqrNorm_g1));

                      if (cos_01 < cosine_threshold)
                      {
                          if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to small cosine(g0,g1) = " << cos_01 << endl;
                          break;
                      } // end if

                      if (sqrNorm_g1 > 1e-12)
                      {
                          scale1 = this->get_scale1(beta, sqrNorm_g0, sqrNorm_g1, gInnerProd);
                          double const s1 = beta*scale1;

                          /* for (int k = 0; k < g0.size(); ++k) Jt_e[k] = -(beta1*g0[k] + s1*g1[k]); */
                          for (int i = 0; i < _countA; i++){
                              for (int j = 0; j < _dimensionA; j++){
                                  _Jt_eA[i][j] = beta1 * Jt_eA0[i][j] + s1 * Jt_eA1[i][j];
                              }
                          }
                          for (int i = 0; i < _countB; i++){
                              for (int j = 0; j < _dimensionB; j++){
                                  _Jt_eB[i][j] = beta1 * Jt_eB0[i][j] + s1 * Jt_eB1[i][j];
                              }
                          }

                          if (!only_distort_gradient)
                          {
                              for (int obj = 0; obj < nObjs; ++obj)
                              {
                                  Vector<double> const& weights0 = cached_weights0[obj];
                                  Vector<double>      & weights1 = cached_weights1[obj];

                                  int const K = weights0.size();
                                  for (int k = 0; k < K; ++k) weights1[k] = beta1*weights0[k] + s1*weights1[k];
                              } // end for (obj)
                          }
                      }
                      else
                      {
                          if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to small |g1| = " << sqrt(sqrNorm_g1) << endl;
                          break;
                      }

                  }else {
                      this->evalJt_e(cached_weights0);
                  }

                  double const norm_Jt_e = norm_L2(_Jt_eA) + norm_L2(_Jt_eB);

                  if (only_distort_gradient)
                     this->fillHessian(cached_weights0);
                  else
                     this->fillHessian(cached_weights1);

                  this->addDamping();
                  this->prepare_schur();
                  this->fillJt_e_schur();
                  this->fill_M();
                  this->solveJtJ();
                  ++this->currentIteration;
                  this->saveAllParameters();

                  this->updateParameters(0, _deltaA);
                  this->updateParameters(1, _deltaB);
                  this->finishUpdateParameters();

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
                  /* double const Psi1_k  = this->eval_level_cost(level, cached_errors); */
                  /* bool success_decrease = (Psi1_k < Psi0_k); */

                  bool stopping_reached, weak_decrease, strong_decrease;

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);

                  // Actutal costs
                  double const Psi1_k  = this->eval_level_cost(level, cached_errors);
                  double const Psi1_k0 = (level > 0) ? this->eval_level_cost(level0, cached_errors) : Psi1_k;

                  double deltaNorm;
                  for (int i = 0; i < _countA; i++) deltaNorm += sqrNorm_L2(_deltaA[i]);
                  for (int i = 0; i < _countB; i++) deltaNorm += sqrNorm_L2(_deltaB[i]);
                  double innerDeltaGrad = inner_product(_deltaA, _Jt_eA) + inner_product(_deltaB, _Jt_eB);

                  double const model_gain = 0.5*(damping_value*deltaNorm + innerDeltaGrad);
                  double const true_gain  = beta1*(Psi0_k0 - Psi1_k0) + beta*scale1*(Psi0_k - Psi1_k);
                  double rho = true_gain / std::max(1e-6, model_gain);

                  weak_decrease   = (Psi1_k0 < Psi0_k0);
                  strong_decrease = (Psi1_k < Psi0_k) && weak_decrease;

                  if (weak_decrease && level > 0)
                  {
                      double const improvementThreshold = 1e-6;
                      double const relImprovement = fabs((Psi1_k0 < Psi0_k0) / Psi0_k0);
                      if (relImprovement < improvementThreshold)
                      {
                          if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to rel. improvement = " << relImprovement << endl;
                          break;
                      }
                  }


                  if (optimizerVerbosenessLevel >= 1)
                  {
                      double const current_cost = this->eval_level_cost(0, cached_errors);
                      // if (optimizerVerbosenessLevel == 1 && (1 || success_decrease))
                      //    cout << "Fast_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost << " lambda = " << damping_value << endl;
                      if (optimizerVerbosenessLevel >= 1 && (0 || weak_decrease))
                      {
                          cout << "Fast_HOM_Optimizer: iteration " << setw(3) << currentIteration << " level = " << level << " new cost = " << setw(12) << current_cost
                             << " Psi_k(prev) = " << setw(12) << Psi0_k << " Psi_k(new) = " << setw(12) << Psi1_k
                              << " success_decreas = " << int(weak_decrease) << " lambda = " << damping_value << endl;


                      }
                  } //end if optimize verbosenessLEvel

                  if (weak_decrease)
                  {
                     damping_value = std::max(damping_value_min, damping_value / 10);
                  }
                  else
                  {
                     this->restoreAllParameters();
                     damping_value *= 10;
                  }

                  /* if (stopping_reached) break; */

               }//end for iter

            } // for nlevel

         }//end void minimize()


    };//end struct definition



};//end namespace

#endif

