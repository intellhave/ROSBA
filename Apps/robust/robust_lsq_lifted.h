// -*- C++ -*-
#ifndef ROBUST_LSQ_LIFTED_H
#define ROBUST_LSQ_LIFTED_H

#include "Base/v3d_timer.h"
#include "robust_lsq_common.h"

#include <random>

namespace Robust_LSQ
{

   constexpr bool use_optimal_weights_init = 0;
   constexpr bool use_weights_EPI = 0;
   constexpr bool use_lifted_varpro = 0;

   /* //constexpr double initial_lifted_weight = 0.5; */
   constexpr double initial_lifted_weight = 1.0;

   constexpr double damping_factor_weights = 1.0;

   struct Quadratic_Parametrization;
   struct Exponential_Parametrization;
   struct Sigmoid_Parametrization;

   typedef Quadratic_Parametrization Default_Paramtrization;
   //typedef Exponential_Parametrization Default_Paramtrization;
   //typedef Sigmoid_Parametrization Default_Paramtrization;

//**********************************************************************

   struct Lifted_Smooth_Truncated
   {
         Lifted_Smooth_Truncated(double const tau)
            : _tau(tau), _tau2(tau*tau)
         { }

         double psi(double const r2)        const { double const w = std::max(0.0, 1.0 - r2/_tau2); return 0.25*_tau2*(1.0-w*w); }
         double gamma(double const w)       const { return 0.25*_tau2*sqr(w-1.0); }
         double dgamma_dw(double const w)   const { return 0.5*_tau2*(w-1.0); }
         double d2gamma_dw2(double const w) const { return 0.5*_tau2; }
         double d3gamma_dw3(double const w) const { return 0; }

         double weight_fun(double const r2) const { return std::max(0.0, 1.0 - r2/_tau2); }
         double weight_fun_deriv(double const r2) const { return (r2 <= 1.0) ? -2.0*sqrt(r2) : 0.0; }

         double d2weight_dr2_at0()          const { return -2.0/_tau2; }

         double const _tau, _tau2;
   }; // end struct lifted_smooth_truncated

   struct Lifted_Welsch
   {
         Lifted_Welsch(double const tau)
            : _tau(tau), _tau2(tau*tau)
         { }

         double psi(double const r2)        const { return 0.5*_tau2*(1.0 - exp(-r2/_tau2)); }
         double gamma(double const w)       const { return 0.5*_tau2 * ((w > 1e-6) ? (1.0 + w*log(w) - w) : 1.0); }
         double dgamma_dw(double const w)   const { return 0.5*_tau2*log(w); }
         double d2gamma_dw2(double const w) const { return 0.5*_tau2/w; }
         double d3gamma_dw3(double const w) const { return -0.5*_tau2/sqr(w); }

         double weight_fun(double const r2) const { return exp(-r2/_tau2); }
         double weight_fun_deriv(double r2) const { return -2.0*sqrt(r2)/_tau2*exp(-r2); }

         double d2weight_dr2_at0()          const { return -2.0/_tau2; }

         double const _tau, _tau2;
   }; // end struct Lifted_Welsch

   struct Lifted_Geman
   {
         Lifted_Geman(double const tau)
            : _tau(tau), _tau2(tau*tau), _tau4(sqr(_tau2))
         { }

         double psi(double const r2)        const { return 0.5*_tau2*r2/(_tau2 + r2); }
         double gamma(double const w)       const { return 0.5*_tau2*sqr(sqrt(w)-1.0); }
         double dgamma_dw(double const w)   const { return 0.5*_tau2*(1.0-1.0/sqrt(w)); }
         double d2gamma_dw2(double const w) const { return 0.25*_tau2/w/sqrt(w); }
         double d3gamma_dw3(double const w) const { return -0.375*_tau2/sqr(w)/sqrt(w); }

         double weight_fun(double const r2) const { return _tau4/sqr(r2 + _tau2); }
         double weight_fun_deriv(double r2) const { return -4.0*_tau4*sqrt(r2)/sqr(_tau2+r2)/(_tau2+r2); }

         double d2weight_dr2_at0()          const { return -4.0/_tau2; }

         double const _tau, _tau2, _tau4;
   }; // end struct Lifted_Geman

//**********************************************************************

   struct Quadratic_Parametrization
   {
         static double   w(double const u)        { return u*u; }
         static Vector3d w_dw_d2w(double const u) { return Vector3d(u*u, 2.0*u, 2.0); }
         static double   d2w0()                   { return 2.0; }
         static double   u(double const w)        { return sqrt(std::max(0.0, w)); }
   }; // end struct Sigmoid_Parametrization

   struct Exponential_Parametrization
   {
         static double w(double const u)          { return exp(u); }
         static Vector3d w_dw_d2w(double const u) { double w = exp(u); return Vector3d(w, w, w); }
         static double d2w0()                     { return 1.0; }
         static double u(double const w)          { return log(std::max(1e-10, w)); }
   }; // end struct Exponential_Parametrization

   struct Sigmoid_Parametrization
   {
         static double w(double const x)
         {
            if (x >= 0.0)
            {
               double const ex = exp(-x); return 1.0 / (1.0 + ex);
            }
            else
            {
               double const ex = exp(x); return ex / (1.0 + ex);
            }
         }

         static Vector3d w_dw_d2w(double const x)
         {
            if (x >= 0)
            {
               double const ex = exp(-x), denom = sqr(1.0+ex)*(1.0+ex);
               return Vector3d(1.0 / (1.0 + ex), ex/sqr(1.0 + ex), ex*(ex-1.0)/denom);
            }
            else
            {
               double const ex = exp(x), denom = sqr(1.0+ex)*(1.0+ex);
               return Vector3d(ex / (1.0 + ex), ex/sqr(1.0 + ex), ex*(1.0-ex)/denom);
            }
         }

         static double d2w0()      { return 0.0; }
         //static double u(double w) { w = std::max(1e-9, std::min(1.0-1e-9, w)); return log(w/(1.0-w)); }
         static double u(double w) { double const th = 1e-10, w0 = std::max(th, w), w1 = std::max(th, 1.0-w); return log(w0/w1); }
   }; // end struct Sigmoid_Parametrization

//**********************************************************************

   template <typename Lifted_Kernel>
   struct Lifted_Optimizer_Base : public NLSQ_LM_Optimizer
   {
         Lifted_Optimizer_Base(NLSQ_ParamDesc const& paramDesc, std::vector<NLSQ_CostFunction *> const& costFunctions,
                               std::vector<Lifted_Kernel> const& kernels, std::vector<double> const& weighting)
            : NLSQ_LM_Optimizer(paramDesc, costFunctions),
              _kernels(kernels), _weighting(weighting),
              _u_params(costFunctions.size()), _u_saved(costFunctions.size()),
              _cached_errors(costFunctions.size()), _cached_weights(costFunctions.size()),
              _cached_alphas(costFunctions.size()), _cached_betas(costFunctions.size()), _cached_deltas(costFunctions.size())
         {
            int const nObjs = _costFunctions.size();

            for (int obj = 0; obj < nObjs; ++obj) _u_params[obj].newsize(_costFunctions[obj]->_nMeasurements);

            for (int obj = 0; obj < nObjs; ++obj) _u_saved[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_errors[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_weights[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_alphas[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_betas[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_deltas[obj].newsize(_costFunctions[obj]->_nMeasurements);
         }

         vector<Lifted_Kernel> const _kernels;
         vector<double>        const _weighting;
         vector<Vector<double> > _u_params, _u_saved, _cached_errors, _cached_weights;
         vector<Vector<double> > _cached_alphas, _cached_betas, _cached_deltas;

         virtual void copyToAllParameters(double * dst) = 0;
         virtual void copyFromAllParameters(double const * src) = 0;

         void cache_errors(NLSQ_CostFunction& costFun, NLSQ_Residuals &residuals, Vector<double> &errors) const
         {
            costFun.preIterationCallback();
            costFun.initializeResiduals();
            costFun.evalAllResiduals(residuals._residuals);

            for (int k = 0; k < costFun._nMeasurements; ++k) errors[k] = sqrNorm_L2(residuals._residuals[k]);
         }

         void fillHessian()
         {
            // Set Hessian to zero
            _hessian.setZero();

            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               
               int const nParamTypes = costFun._usedParamTypes.size();
               int const nMeasurements = costFun._nMeasurements, residualDim = costFun._measurementDimension;

               Matrix<double> C(residualDim, residualDim);

               MatrixArray<int> const& index = *_hessianIndices[obj];
               NLSQ_Residuals   const& residuals = *_residuals[obj];

               Vector<double> const& errors  = _cached_errors[obj];
               Vector<double> const& weights = _cached_weights[obj];
               Vector<double> const& alphas  = _cached_alphas[obj];
               Vector<double> const& betas   = _cached_betas[obj];
               Vector<double> const& deltas  = _cached_deltas[obj];
               double         const  W       = _weighting[obj];

               vector<int> const& usedParamTypes = costFun._usedParamTypes;

               for (int i1 = 0; i1 < nParamTypes; ++i1)
               {
                  int const t1 = usedParamTypes[i1], dim1 = _paramDesc.dimension[t1];

                  MatrixArray<double> const& Js1 = *residuals._Js[i1];

                  for (int i2 = 0; i2 < nParamTypes; ++i2)
                  {
                     int const t2 = usedParamTypes[i2], dim2 = _paramDesc.dimension[t2];

                     MatrixArray<double> const& Js2 = *residuals._Js[i2];

                     Matrix<double> J1tJ2(dim1, dim2), C_J2(residualDim, dim2);

                     // Ignore non-existent Hessians lying in the lower triangular part.
                     if (!_hessian.Hs[t1][t2]) continue;

                     MatrixArray<double>& Hs = *_hessian.Hs[t1][t2];

                     for (int k = 0; k < nMeasurements; ++k)
                     {
                        int const ix1 = costFun._correspondingParams[k][i1];
                        int const id1 = this->getParamId(t1, ix1);
                        int const ix2 = costFun._correspondingParams[k][i2];
                        int const id2 = this->getParamId(t2, ix2);

#if 1 || defined(ONLY_UPPER_TRIANGULAR_HESSIAN)
                        if (id1 > id2) continue; // only store the upper diagonal blocks
#endif
                        int const n = index[k][i1][i2];
                        assert(n < Hs.count());

                        //makeZeroMatrix(C);
                        makeOuterProductMatrix(residuals._residuals[k], C);
                        if (1 || !use_weights_EPI)
                           scaleMatrixIP(-sqr(alphas[k])/betas[k], C);
                        else
                           scaleMatrixIP(-alphas[k], C);

                        double const w = weights[k];
                        for (int j = 0; j < C.num_rows(); ++j) C[j][j] += w;
                        scaleMatrixIP(W, C);

                        multiply_A_B(C, Js2[k], C_J2);
                        multiply_At_B(Js1[k], C_J2, J1tJ2);
                        addMatricesIP(J1tJ2, Hs[n]);
                     } // end for (k)
                  } // end for (i2)
               } // end for (i1)
            } // end for (obj)
         } // end fillHessian()

         void evalJt_e(Vector<double>& Jt_e)
         {
            makeZeroVector(Jt_e);

            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               NLSQ_Residuals const& residuals = *_residuals[obj];

               Vector<double> const& weights = _cached_weights[obj];
               Vector<double> const& alphas  = _cached_alphas[obj];
               Vector<double> const& betas   = _cached_betas[obj];
               Vector<double> const& deltas  = _cached_deltas[obj];
               double         const  W       = _weighting[obj];

               int const nParamTypes = costFun._usedParamTypes.size(), nMeasurements = costFun._nMeasurements;

               for (int i = 0; i < nParamTypes; ++i)
               {
                  int const paramType = costFun._usedParamTypes[i], paramDim = _paramDesc.dimension[paramType];

                  MatrixArray<double> const& J = *residuals._Js[i];

                  Vector<double> Jkt_e(paramDim);

                  for (int k = 0; k < nMeasurements; ++k)
                  {
                     int const id = costFun._correspondingParams[k][i];
                     int const dstRow = _paramTypeRowStart[paramType] + id*paramDim;

                     multiply_At_v(J[k], residuals._residuals[k], Jkt_e);
                     if (1 || !use_weights_EPI)
                     {
                        double const s = W * (weights[k] - alphas[k]*deltas[k]/betas[k]);
                        scaleVectorIP(s, Jkt_e);
                     }
                     else
                     {
                        double const s = W * weights[k];
                        scaleVectorIP(s, Jkt_e);
                     } // end if
                     for (int l = 0; l < paramDim; ++l) Jt_e[dstRow + l] += Jkt_e[l];
                  } // end for (k)
               } // end for (i)
            } // end for (obj)
         } // end evalJt_e()

         bool solve_JtJ(Vector<double> &Jt_e, Vector<double> &deltaPerm, Vector<double> &delta)
         {
            bool success_LDL = true;

            int const nCols = _JtJ_Parent.size();
            //int const nnz   = _JtJ.getNonzeroCount();
            int const lnz   = _JtJ_Lp.back();

            vector<int> Li(lnz);
            vector<double> Lx(lnz);
            vector<double> D(nCols), Y(nCols);
            vector<int> workPattern(nCols), workFlag(nCols);

            int * colStarts = (int *)_JtJ.getColumnStarts();
            int * rowIdxs   = (int *)_JtJ.getRowIndices();
            double * values = _JtJ.getValues();

            int const d = LDL_numeric(nCols, colStarts, rowIdxs, values,
                                      &_JtJ_Lp[0], &_JtJ_Parent[0], &_JtJ_Lnz[0],
                                      &Li[0], &Lx[0], &D[0],
                                      &Y[0], &workPattern[0], &workFlag[0]);

            if (d == nCols)
            {
               LDL_perm(nCols, &deltaPerm[0], &Jt_e[0], &_perm_JtJ[0]);
               LDL_lsolve(nCols, &deltaPerm[0], &_JtJ_Lp[0], &Li[0], &Lx[0]);
               LDL_dsolve(nCols, &deltaPerm[0], &D[0]);
               LDL_ltsolve(nCols, &deltaPerm[0], &_JtJ_Lp[0], &Li[0], &Lx[0]);
               LDL_permt(nCols, &delta[0], &deltaPerm[0], &_perm_JtJ[0]);
            }
            else
            {
               if (optimizerVerbosenessLevel >= 2)
               {
                  cout << "Lifted_Optimizer_Base: LDL decomposition failed with d = " << d << ". Increasing lambda." << endl;
               }
               success_LDL = false;
            }
            return success_LDL;
         } // end solve_JtJ()
   }; // end struct Lifted_Optimizer_Base

//**********************************************************************

   template <typename Lifted_Kernel, bool use_Gauss_Newton = true, typename Parametrization = Default_Paramtrization>
   struct Lifted_Optimizer : public Lifted_Optimizer_Base<Lifted_Kernel>
   {
         typedef Lifted_Optimizer_Base<Lifted_Kernel> Base;

         using Base::_costFunctions;
         using Base::_residuals;
         using Base::_u_params;
         using Base::_u_saved;
         using Base::_cached_errors;
         using Base::_cached_weights;
         using Base::_cached_alphas;
         using Base::_cached_betas;
         using Base::_cached_deltas;

         using Base::_kernels;
         using Base::_weighting;

         using Base::currentIteration;
         using Base::maxIterations;
         using Base::status;

         using Base::_hessian;
         using Base::_paramDesc;
         using Base::_paramTypeRowStart;

         Lifted_Optimizer(NLSQ_ParamDesc const& paramDesc, std::vector<NLSQ_CostFunction *> const& costFunctions,
                          std::vector<Lifted_Kernel> const& kernels, std::vector<double> const& weighting)
            : Lifted_Optimizer_Base<Lifted_Kernel>(paramDesc, costFunctions, kernels, weighting)
         {
            int const nObjs = this->_costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj) fillVector(Parametrization::u(initial_lifted_weight), this->_u_params[obj]);
         }

         double eval_robust_objective(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _costFunctions.size();

            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto const errors = cached_errors[obj];
               auto const kernel = _kernels[obj];
               double const W    = _weighting[obj];
               for (int k = 0; k < errors.size(); ++k) cost += W * kernel.psi(errors[k]);
            }
            return cost;
         }

         double eval_robust_objective()
         {
            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

            return this->eval_robust_objective(_cached_errors);
         }

         double eval_lifted_cost(vector<Vector<double> > const& cached_errors) const
         {
            double cost1 = 0, cost2 = 0;
            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto const errors = cached_errors[obj];
               auto const kernel = _kernels[obj];
               double const W    = _weighting[obj];
#if 0
               for (int k = 0; k < errors.size(); ++k) cost += W * kernel.psi(errors[k]);
#else
               auto const& u = _u_params[obj];
               for (int k = 0; k < errors.size(); ++k)
               {
                  double const w = Parametrization::w(u[k]);
                  cost1 += W * 0.5*w*errors[k];
                  cost2 += W * kernel.gamma(w);
               }
#endif
            }
            //cout << "cost1 = " << cost1 << " cost2 = " << cost2 << endl;
            return cost1 + cost2;
         }

         void minimize()
         {
            std::default_random_engine rng;
            std::normal_distribution<double> dist(0.0, 1e-6);

            this->status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (this->_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: exiting since d.o.f is zero." << endl;
               this->status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = this->_JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _costFunctions.size();

            double damping_value = this->tau;

            int LDL_failures = 0;

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            Timer t("HQ");
            ofstream log_file("log_HQ.txt");
            double bestCost = 1e50; //Best cost of from previous iteration;
            t.start();
            for (currentIteration = 0; currentIteration < maxIterations; ++this->currentIteration)
            {
               for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

               if (use_optimal_weights_init && currentIteration == 0)
               {
                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     Vector<double> &us = _u_params[obj];
                     Vector<double> const& errors = _cached_errors[obj];
                     auto const kernel = _kernels[obj];
                     int const K = us.size();
#if 1
                     double const eta = 0.00001;
                     for (int k = 0; k < K; ++k) us[k] = Parametrization::u((1.0-eta)*kernel.weight_fun(errors[k]) + eta);
#else
                     for (int k = 0; k < K; ++k) us[k] = Parametrization::u(1.0/sqrt(1.0 + 100.0*errors[k]));
#endif
                     for (int k = 0; k < K; ++k)
                     {
                        double const u = us[k];
                        if (isnan(u) || !isfinite(u)) cout << " k = " << k << " us[k] = " << u << " r2 = " << errors[k] << endl;
                     }
                  } // end for (obj)
               }

               if (use_weights_EPI)
               {
                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     Vector<double> &us = _u_params[obj];
                     Vector<double> const& errors = _cached_errors[obj];
                     auto const            kernel = _kernels[obj];

                     int const K = us.size();
                     for (int k = 0; k < K; ++k) us[k] = Parametrization::u(kernel.weight_fun(errors[k]));
                  } // end for (obj)
               } // end if

               if (use_lifted_varpro)
               {
                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     Vector<double> &us = _u_params[obj];
                     Vector<double> const& errors = _cached_errors[obj];
                     auto const            kernel = _kernels[obj];

                     int const K = us.size();
                     for (int k = 0; k < K; ++k) us[k] = sqrt(std::max(0.0, kernel.weight_fun(errors[k]))) + 1.0*dist(rng);
                     //for (int k = 0; k < K; ++k) us[k] = std::max(1e-3, us[k]);
                  } // end for (obj)
               } // end if

               double const initial_cost = use_lifted_varpro ? this->eval_robust_objective(_cached_errors) : this->eval_lifted_cost(_cached_errors);
               double init_robust_cost = this->eval_robust_objective(_cached_errors);
               bestCost = min(bestCost, init_robust_cost);
               t.stop();
               if (bestCost < 1e20)
                     log_file  << t.getTime() << "\t" << currentIteration << "\t" << bestCost << endl;                          
               t.start();
               if (optimizerVerbosenessLevel >= 2)
               {
                  cout << "Lifted_Optimizer: iteration: " << currentIteration << ", initial |residual|^2 = " << initial_cost << " lambda = " << damping_value << endl;                  

               }

               this->fillJacobians();

               for (int obj = 0; obj < nObjs; ++obj)
               {
                  NLSQ_CostFunction   & costFun   = *_costFunctions[obj];
                  NLSQ_Residuals const& residuals = *_residuals[obj];
                  Vector<double> const& us        = _u_params[obj];
                  Vector<double> const& errors    = _cached_errors[obj];
                  Vector<double>      & weights   = _cached_weights[obj];
                  Vector<double>      & alphas    = _cached_alphas[obj];
                  Vector<double>      & betas     = _cached_betas[obj];
                  Vector<double>      & deltas    = _cached_deltas[obj];
                  auto const            kernel    = _kernels[obj];

                  double const d2gamma1 = kernel.d2gamma_dw2(1.0), d3gamma1 = kernel.d3gamma_dw3(1.0);

                  int const K = costFun._nMeasurements;
                  for (int k = 0; k < K; ++k)
                  {
                     if (use_lifted_varpro)
                     {
                        double const u = us[k], w = sqr(u), dw = 2*u, r2 = errors[k], dgamma_dw = kernel.dgamma_dw(w);

                        weights[k] = w;
                        //deltas[k]  = 0.0;
                        deltas[k]  = dw*(0.5*r2 + dgamma_dw);
                        alphas[k]  = 0.5*dw;

                        double hessian = 0;
#if 0
                        double const denom1 = kernel.gamma(w);
                        if (fabs(denom1) > 1e-6)
                           hessian += 0.5*sqr(dgamma_dw)/denom1;
                        else
                           hessian += 0.5*(2*d2gamma1 + (w-1.0)*1.3333333*d3gamma1);

                        hessian -= 0.5 * ((fabs(w) > 1e-6) ? dgamma_dw/w : d2gamma1);
#else
                        hessian += r2;
                        double const denom1 = kernel.gamma(w);
                        if (fabs(denom1) > 1e-6)
                           hessian += 0.5*w*sqr(dgamma_dw)/denom1;
                        else
                           hessian += 0.5*w*(2*d2gamma1 + (w-1.0)*1.3333333*d3gamma1);
#endif

                        betas[k] = hessian + 1e-6;
                     }
                     else
                     {
                        double const u = us[k];
                        Vector3d const vals = Parametrization::w_dw_d2w(u);
                        double const w = vals[0], dw = vals[1], d2w = vals[2], d2w0 = Parametrization::d2w0();
                        double const r2 = errors[k], dgamma_dw = kernel.dgamma_dw(w);

                        weights[k] = w;
                        //deltas[k]  = dw * (0.5*r2 + dgamma_dw);
                        deltas[k]  = use_weights_EPI ? 0.0 : (dw * (0.5*r2 + dgamma_dw));
                        alphas[k]  = use_Gauss_Newton ? 0.5*dw : dw;

                        if (use_Gauss_Newton)
                        {
                           // The Gauss-Newton approximation
                           if (use_weights_EPI)
                           {
                              double hessian = r2 * (fabs(w) > 1e-4 ? 0.25*sqr(dw)/w : 0.5*d2w0);
                              double const denom = kernel.gamma(w);
                              if (fabs(denom) > 1e-4)
                                 hessian += 0.5*sqr(dw * dgamma_dw)/denom;
                              else
                                 hessian += 0.5*sqr(dw) * (2*d2gamma1 + (w-1.0)*1.3333333*d3gamma1);
                              betas[k] = hessian + 1e-10;
                           }
                           else
                           {
                              double hessian = r2 * (fabs(w) > 1e-4 ? 0.25*sqr(dw)/w : 0.5*d2w0);
                              double const denom = kernel.gamma(w);
                              if (fabs(denom) > 1e-4)
                                 hessian += 0.5*sqr(dw * dgamma_dw)/denom;
                              else
                                 hessian += 0.5*sqr(dw) * (2*d2gamma1 + (w-1.0)*1.3333333*d3gamma1);
                              betas[k] = hessian + damping_factor_weights*damping_value;
                           } // end if
                        }
                        else
                        {
                           // The Newton approximation
                           double hessian = r2 * ((fabs(w) > 1e-4) ? sqr(dw)/w : 2.0*d2w0);
                           hessian = std::max(hessian, d2w*(0.5*r2 + dgamma_dw) + sqr(dw)*kernel.d2gamma_dw2(w));
                           betas[k] = hessian + damping_value;
                        } // endif
                     } // end if
                  } // end for (k)
               } // end for (obj)

               //displayVector_n(_cached_betas[0], 20);
               //displayVector_n(_cached_deltas[0], 50);

               this->evalJt_e(Jt_e); // Jt_e holds omega(e) J^T e at this point

               double const norm_Linf_Jt_e = norm_Linf(Jt_e);

               if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
               {
                  if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: converged due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
                  break;
               }

               scaleVectorIP(-1.0, Jt_e);

               this->fillHessian();
               bool success_LDL = false;

               // Augment the diagonals
               for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
               {
                  MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
                  vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
                  int const dim = Hs.num_cols(), count = Hs.count();

                  // Only augment those with the same parameter id
                  for (int n = 0; n < count; ++n)
                  {
                     if (nzPairs[n].first != nzPairs[n].second) continue;
                     for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
                  }
               } // end for (paramType)

               this->fillJtJ();
               success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
               if (!success_LDL) ++LDL_failures;

               this->copyToAllParameters(&x_saved[0]);
               for (int obj = 0; obj < nObjs; ++obj) _u_saved[obj] = _u_params[obj];

               bool success_decrease = false;
               if (success_LDL)
               {
                  double const deltaSqrLength = sqrNorm_L2(delta);
                  double const paramLength = this->getParameterLength();

                  if (optimizerVerbosenessLevel >= 3)
                     cout << "Lifted_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                  if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: converged due to small update, deltaSqrLength = " << deltaSqrLength << endl;
                     break;
                  }

                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     int const paramDim = _paramDesc.dimension[paramType];
                     int const count    = _paramDesc.count[paramType];
                     int const rowStart = _paramTypeRowStart[paramType];

                     VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                     this->updateParameters(paramType, deltaParam);
                  } // end for (paramType)
                  this->finishUpdateParameters();

                  if (!use_weights_EPI)
                  {
                     // Now update u
                     double norm_du = 0;
                     for (int obj = 0; obj < nObjs; ++obj)
                     {
                        NLSQ_CostFunction& costFun   = *_costFunctions[obj];
                        NLSQ_Residuals&    residuals = *_residuals[obj];

                        Vector<double>& u = _u_params[obj];

                        Vector<double> const& alphas  = _cached_alphas[obj];
                        Vector<double> const& betas   = _cached_betas[obj];
                        Vector<double> const& deltas  = _cached_deltas[obj];

                        int const nParamTypes = costFun._usedParamTypes.size(), K = costFun._nMeasurements;

                        vector<double> delta_Jt_e(K, 0.0);

                        for (int i = 0; i < nParamTypes; ++i)
                        {
                           int const paramType = costFun._usedParamTypes[i];
                           int const paramDim = _paramDesc.dimension[paramType];
                           int const rowStart = _paramTypeRowStart[paramType];

                           double * deltaParam = &delta[0] + rowStart;

                           MatrixArray<double> const& J = *residuals._Js[i];

                           // Accumulate e^T J delta
                           Vector<double> Jkt_e(paramDim);
                           for (int k = 0; k < K; ++k)
                           {
                              int const id = costFun._correspondingParams[k][i];
                              Vector<double> delta_k(paramDim, deltaParam + paramDim*id);

                              multiply_At_v(J[k], residuals._residuals[k], Jkt_e);
                              delta_Jt_e[k] += innerProduct(delta_k, Jkt_e);
                           } // end for (k)
                        } // end for (i)

                        //cout << "u(before) = "; displayVector_n(u, 20);

                        for (int k = 0; k < K; ++k)
                        {
                           double const du = -(deltas[k] + alphas[k]*delta_Jt_e[k]) / betas[k];
                           u[k] += du;
                           //if (fabs(u[k]) > 1.0) u[k] = (u[k] > 0) ? 1.0 : -1.0;
                           norm_du += sqr(du) / K;
                        } // end for (k)
                        //cout << "u(after) = "; displayVector_n(u, 20);
                     } // end for (obj)
                     //cout << " RMS(du) = " << sqrt(norm_du) << endl;
                  } // end if

                  // Check if new cost is better than best one at current level
                  for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

                  if (use_weights_EPI)
                  {
                     for (int obj = 0; obj < nObjs; ++obj)
                     {
                        Vector<double> &us = _u_params[obj];
                        Vector<double> const& errors = _cached_errors[obj];
                        auto const            kernel = _kernels[obj];

                        int const K = us.size();
                        for (int k = 0; k < K; ++k) us[k] = Parametrization::u(kernel.weight_fun(errors[k]));
                     } // end for (obj)
                  } // end if

                  double const actual_cost  = this->eval_robust_objective(_cached_errors);
                  double const current_cost = use_lifted_varpro ? actual_cost : this->eval_lifted_cost(_cached_errors);
                                    
                  success_decrease = (current_cost < initial_cost);
                  if (optimizerVerbosenessLevel >= 1 && success_decrease)
                  {
                     cout << "Lifted_Optimizer: iteration: " << setw(3) << currentIteration << " previous lifted cost = " << setw(12) << initial_cost
                          << " new lifted cost = " << setw(12) << current_cost << " actual robust cost = " << setw(12) << actual_cost << " lambda = " << damping_value << endl;

                     if (actual_cost < bestCost)
                        bestCost = actual_cost;                                          
                  }

                  
                  if (optimizerVerbosenessLevel >= 2 && !success_decrease) cout << "Lifted_Optimizer: iteration: " << currentIteration << " previous lifted cost = " << initial_cost
                                                                                << " new lifted cost = " << current_cost << " actual robust cost = " << actual_cost << " lambda = " << damping_value << endl;
               } // end if (success_LDL)

               if (success_decrease)
               {
                  damping_value = std::max(damping_value_min, damping_value / 10);
               }
               else
               {
                  if (success_LDL)
                  {
                     this->copyFromAllParameters(&x_saved[0]);
                     for (int obj = 0; obj < nObjs; ++obj) _u_params[obj] = _u_saved[obj];
                  }
                  damping_value *= 10;
               }
            } // end for (currentIteration)
            log_file.close();

            if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
            {
               cout << "Lifted_Optimizer: reached maximum number of iterations, exiting." << endl;
            }

            if (optimizerVerbosenessLevel >= 2)
               cout << "Leaving Lifted_Optimizer::minimize(): LDL_failures = " << LDL_failures << endl;
         } // end minimize()
   }; // end struct Lifted_Optimizer

//**********************************************************************

//    template <typename Lifted_Kernel>
//    struct Natural_Lifted_Optimizer : public NLSQ_LM_Optimizer
//    {
//          typedef Lifted_Optimizer_Base<Lifted_Kernel> Base;

//          using Base::_costFunctions;
//          using Base::_residuals;
//          using Base::_u_params;
//          using Base::_u_saved;
//          using Base::_cached_errors;
//          using Base::_cached_weights;
//          using Base::_cached_alphas;
//          using Base::_cached_betas;
//          using Base::_cached_deltas;

//          using Base::_kernels;
//          using Base::_weighting;

//          using Base::currentIteration;
//          using Base::maxIterations;
//          using Base::status;

//          using Base::_hessian;
//          using Base::_paramDesc;
//          using Base::_paramTypeRowStart;

//          Natural_Lifted_Optimizer(NLSQ_ParamDesc const& paramDesc, std::vector<NLSQ_CostFunction *> const& costFunctions,
//                                   std::vector<Lifted_Kernel> const& kernels, std::vector<double> const& weighting)
//             : Lifted_Optimizer_Base<Lifted_Kernel>(paramDesc, costFunctions, kernels, weighting)
//          {
//             int const nObjs = _costFunctions.size();
//             for (int obj = 0; obj < nObjs; ++obj) fillVector(0.0, _u_params[obj]);
//          }

//          double eval_robust_objective()
//          {
//             int const nObjs = _costFunctions.size();
//             for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);
//             return this->eval_current_cost(_cached_errors);
//          }

//          double eval_current_cost(vector<Vector<double> > const& cached_errors) const
//          {
//             double cost = 0;
//             int const nObjs = this->_costFunctions.size();
//             for (int obj = 0; obj < nObjs; ++obj)
//             {
//                auto const errors = cached_errors[obj];
//                auto const kernel = _kernels[obj];
//                double const W    = _weighting[obj];
// #if 0
//                for (int k = 0; k < errors.size(); ++k) cost += W * kernel.psi(errors[k]);
// #else
//                auto const& u = _u_params[obj];
//                for (int k = 0; k < errors.size(); ++k)
//                {
//                   double const z = 0.5*sqr(u[k]);
//                   cost += W * (kernel.psi(u[k]) + 0.5*kernel.weight_fun(u[k])*(errors[k] - z));
//                }
// #endif
//             }
//             return cost;
//          }


// //**********************************************************************

//          void minimize()
//          {
//             this->status = LEVENBERG_OPTIMIZER_TIMEOUT;

//             if (this->_totalParamCount == 0)
//             {
//                // No degrees of freedom, nothing to optimize.
//                if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: exiting since d.o.f is zero." << endl;
//                this->status = LEVENBERG_OPTIMIZER_CONVERGED;
//                return;
//             }

//             int const totalParamDimension = this->_JtJ.num_cols();

//             vector<double> x_saved(totalParamDimension);

//             Vector<double> Jt_e(totalParamDimension);
//             Vector<double> delta(totalParamDimension);
//             Vector<double> deltaPerm(totalParamDimension);

//             int const nObjs = _costFunctions.size();

//             double damping_value = this->tau;

//             int LDL_failures = 0;

//             for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

//             for (currentIteration = 0; currentIteration < maxIterations; ++this->currentIteration)
//             {
//                for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

//                double const initial_cost = this->eval_current_cost(_cached_errors);

//                if (optimizerVerbosenessLevel >= 1)
//                   cout << "Lifted_Optimizer: iteration: " << currentIteration << ", initial |residual|^2 = " << initial_cost << " lambda = " << damping_value << endl;

//                this->fillJacobians();

//                for (int obj = 0; obj < nObjs; ++obj)
//                {
//                   NLSQ_CostFunction   & costFun   = *_costFunctions[obj];
//                   NLSQ_Residuals const& residuals = *_residuals[obj];
//                   Vector<double> const& us        = _u_params[obj];
//                   Vector<double> const& errors    = _cached_errors[obj];
//                   Vector<double>      & weights   = _cached_weights[obj];
//                   Vector<double>      & alphas    = _cached_alphas[obj];
//                   Vector<double>      & betas     = _cached_betas[obj];
//                   Vector<double>      & deltas    = _cached_deltas[obj];
//                   auto const            kernel    = _kernels[obj];

//                   double const d2gamma1 = kernel.d2gamma_dw2(1.0), d3gamma1 = kernel.d3gamma_dw3(1.0);

//                   int const K = costFun._nMeasurements;
//                   for (int k = 0; k < K; ++k)
//                   {
//                      double const u = us[k], w = kernel.weight_fun(u), dw = kernel.weight_fun_deriv(u);

//                      Vector3d const vals = Parametrization::w_dw_d2w(u);
//                      double const w = vals[0], dw = vals[1], d2w = vals[2], d2w0 = Parametrization::d2w0();
//                      double const r2 = errors[k], dgamma_dw = kernel.dgamma_dw(w);

//                      weights[k] = w;
//                      deltas[k]  = dw * (0.5*r2 + dgamma_dw);
//                      alphas[k]  = use_Gauss_Newton ? 0.5*dw : dw;

//                      if (use_Gauss_Newton)
//                      {
//                         // The Gauss-Newton approximation
//                         double hessian = r2 * (fabs(w) > 1e-4 ? 0.25*sqr(dw)/w : 0.5*d2w0);
//                         double const denom = kernel.gamma(w);
//                         if (fabs(denom) > 1e-4)
//                            hessian += 0.5*sqr(dw * dgamma_dw)/denom;
//                         else
//                            hessian += 0.5*sqr(dw) * (2*d2gamma1 + (w-1.0)*1.3333333*d3gamma1);
//                         betas[k] = hessian + damping_value;
//                      }
//                      else
//                      {
//                         // The Newton approximation
//                         double hessian = r2 * ((fabs(w) > 1e-4) ? sqr(dw)/w : 2.0*d2w0);
//                         hessian = std::max(hessian, d2w*(0.5*r2 + dgamma_dw) + sqr(dw)*kernel.d2gamma_dw2(w));
//                         betas[k] = hessian + damping_value;
//                      } // endif
//                   } // end for (k)
//                } // end for (obj)

//                //displayVector_n(_cached_betas[0], 20);

//                this->evalJt_e(Jt_e); // Jt_e holds omega(e) J^T e at this point

//                double const norm_Linf_Jt_e = norm_Linf(Jt_e);

//                if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
//                {
//                   if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: converged due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
//                   break;
//                }

//                scaleVectorIP(-1.0, Jt_e);

//                this->fillHessian();
//                bool success_LDL = false;

//                // Augment the diagonals
//                for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
//                {
//                   MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
//                   vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
//                   int const dim = Hs.num_cols(), count = Hs.count();

//                   // Only augment those with the same parameter id
//                   for (int n = 0; n < count; ++n)
//                   {
//                      if (nzPairs[n].first != nzPairs[n].second) continue;
//                      for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
//                   }
//                } // end for (paramType)

//                this->fillJtJ();
//                success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
//                if (!success_LDL) ++LDL_failures;

//                this->copyToAllParameters(&x_saved[0]);
//                for (int obj = 0; obj < nObjs; ++obj) _u_saved[obj] = _u_params[obj];

//                bool success_decrease = false;
//                if (success_LDL)
//                {
//                   double const deltaSqrLength = sqrNorm_L2(delta);
//                   double const paramLength = this->getParameterLength();

//                   if (optimizerVerbosenessLevel >= 3)
//                      cout << "Lifted_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

//                   if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
//                   {
//                      if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: converged at current level to small update, deltaSqrLength = " << deltaSqrLength << endl;
//                      break;
//                   }

//                   for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
//                   {
//                      int const paramDim = _paramDesc.dimension[paramType];
//                      int const count    = _paramDesc.count[paramType];
//                      int const rowStart = _paramTypeRowStart[paramType];

//                      VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
//                      this->updateParameters(paramType, deltaParam);
//                   } // end for (paramType)
//                   this->finishUpdateParameters();

//                   // Now update u
//                   double norm_du = 0;
//                   for (int obj = 0; obj < nObjs; ++obj)
//                   {
//                      NLSQ_CostFunction& costFun   = *_costFunctions[obj];
//                      NLSQ_Residuals&    residuals = *_residuals[obj];

//                      Vector<double>& u = _u_params[obj];

//                      Vector<double> const& alphas  = _cached_alphas[obj];
//                      Vector<double> const& betas   = _cached_betas[obj];
//                      Vector<double> const& deltas  = _cached_deltas[obj];

//                      int const nParamTypes = costFun._usedParamTypes.size(), K = costFun._nMeasurements;

//                      vector<double> delta_Jt_e(K, 0.0);

//                      for (int i = 0; i < nParamTypes; ++i)
//                      {
//                         int const paramType = costFun._usedParamTypes[i];
//                         int const paramDim = _paramDesc.dimension[paramType];
//                         int const rowStart = _paramTypeRowStart[paramType];

//                         double * deltaParam = &delta[0] + rowStart;

//                         MatrixArray<double> const& J = *residuals._Js[i];

//                         // Accumulate e^T J delta
//                         Vector<double> Jkt_e(paramDim);
//                         for (int k = 0; k < K; ++k)
//                         {
//                            int const id = costFun._correspondingParams[k][i];
//                            Vector<double> delta_k(paramDim, deltaParam + paramDim*id);

//                            multiply_At_v(J[k], residuals._residuals[k], Jkt_e);
//                            delta_Jt_e[k] += innerProduct(delta_k, Jkt_e);
//                         } // end for (k)
//                      } // end for (i)

//                      //cout << "u(before) = "; displayVector_n(u, 20);

//                      for (int k = 0; k < K; ++k)
//                      {
//                         double const du = -(deltas[k] + alphas[k]*delta_Jt_e[k]) / betas[k];
//                         u[k] += du;
//                         //if (fabs(u[k]) > 1.0) u[k] = (u[k] > 0) ? 1.0 : -1.0;
//                         norm_du += sqr(du) / K;
//                      } // end for (k)
//                      //cout << "u(after) = "; displayVector_n(u, 20);
//                   } // end for (obj)
//                   //cout << " RMS(du) = " << sqrt(norm_du) << endl;

//                   // Check if new cost is better than best one at current level
//                   for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

//                   if (use_weights_EPI)
//                   {
//                      for (int obj = 0; obj < nObjs; ++obj)
//                      {
//                         Vector<double> &us = _u_params[obj];
//                         Vector<double> const& errors = _cached_errors[obj];
//                         auto const            kernel = _kernels[obj];

//                         int const K = us.size();
//                         for (int k = 0; k < K; ++k) us[k] = sqrt(kernel.weight_fun(errors[k]));
//                      } // end for (obj)
//                   } // end if

//                   double const current_cost = this->eval_current_cost(_cached_errors);

//                   if (optimizerVerbosenessLevel >= 2) cout << "Lifted_Optimizer: success_LDL = " << int(success_LDL) << " new cost = " << current_cost << endl;
//                   success_decrease = (current_cost < initial_cost);
//                } // end if (success_LDL)

//                if (success_decrease)
//                {
//                   damping_value = std::max(1e-10, damping_value / 10);
//                }
//                else
//                {
//                   if (success_LDL)
//                   {
//                      this->copyFromAllParameters(&x_saved[0]);
//                      for (int obj = 0; obj < nObjs; ++obj) _u_params[obj] = _u_saved[obj];
//                   }
//                   damping_value *= 10;
//                }
//             } // end for (currentIteration)

//             if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
//             {
//                cout << "Lifted_Optimizer: reached maximum number of iterations, exiting." << endl;
//             }

//             if (optimizerVerbosenessLevel >= 2)
//                cout << "Leaving Lifted_Optimizer::minimize(): LDL_failures = " << LDL_failures << endl;
//          } // end minimize()
//    }; // end struct Natural_Lifted_Optimizer

//**********************************************************************

   template <typename Lifted_Kernel>
   struct Alternate_SqrtPsi_Optimizer : public NLSQ_LM_Optimizer
   {
         Alternate_SqrtPsi_Optimizer(NLSQ_ParamDesc const& paramDesc, std::vector<NLSQ_CostFunction *> const& costFunctions,
                                      std::vector<Lifted_Kernel> const& kernels, std::vector<double> const& weighting)
            : NLSQ_LM_Optimizer(paramDesc, costFunctions),
              _kernels(kernels), _weighting(weighting),
              _cached_errors(costFunctions.size()), _cached_weights(costFunctions.size()), _cached_dws(costFunctions.size()),
              _cached_Cs(costFunctions.size()), _cached_bs(costFunctions.size())
         {
            int const nObjs = _costFunctions.size();

            for (int obj = 0; obj < nObjs; ++obj) _cached_errors[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_weights[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_dws[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_Cs[obj].newsize(_costFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _cached_bs[obj].newsize(_costFunctions[obj]->_nMeasurements);
         }

         vector<Lifted_Kernel> const _kernels;
         vector<double>        const _weighting;
         vector<Vector<double> >   _cached_errors, _cached_weights, _cached_dws;
         vector<Vector<Vector3d> > _cached_Cs;
         vector<Vector<double> >   _cached_bs;

         virtual void copyToAllParameters(double * dst) = 0;
         virtual void copyFromAllParameters(double const * src) = 0;

         void cache_errors(NLSQ_CostFunction& costFun, NLSQ_Residuals &residuals, Vector<double> &errors) const
         {
            costFun.preIterationCallback();
            costFun.initializeResiduals();
            costFun.evalAllResiduals(residuals._residuals);

            for (int k = 0; k < costFun._nMeasurements; ++k) errors[k] = sqrNorm_L2(residuals._residuals[k]);
         }

         void fillJacobians()
         {
            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               NLSQ_Residuals& residuals = *_residuals[obj];
               costFun.initializeJacobian();
               costFun.fillAllJacobians(residuals._weights, residuals._Js);
            } // end for (obj)
         } // end fillJacobians()

         void fillHessian()
         {
            // Set Hessian to zero
            _hessian.setZero();

            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               
               int const nParamTypes = costFun._usedParamTypes.size();
               int const nMeasurements = costFun._nMeasurements, residualDim = costFun._measurementDimension;

               Matrix<double> C(residualDim, residualDim);

               MatrixArray<int> const& index = *_hessianIndices[obj];
               NLSQ_Residuals   const& residuals = *_residuals[obj];

               Vector<double>   const& errors  = _cached_errors[obj];
               Vector<double>   const& weights = _cached_weights[obj];
               Vector<double>   const& dws     = _cached_dws[obj];
               Vector<Vector3d> const& Cs      = _cached_Cs[obj];
               double           const  W       = _weighting[obj];

               vector<int> const& usedParamTypes = costFun._usedParamTypes;

               for (int i1 = 0; i1 < nParamTypes; ++i1)
               {
                  int const t1 = usedParamTypes[i1], dim1 = _paramDesc.dimension[t1];

                  MatrixArray<double> const& Js1 = *residuals._Js[i1];

                  for (int i2 = 0; i2 < nParamTypes; ++i2)
                  {
                     int const t2 = usedParamTypes[i2], dim2 = _paramDesc.dimension[t2];

                     MatrixArray<double> const& Js2 = *residuals._Js[i2];

                     Matrix<double> J1tJ2(dim1, dim2), C_J2(residualDim, dim2);

                     // Ignore non-existent Hessians lying in the lower triangular part.
                     if (!_hessian.Hs[t1][t2]) continue;

                     MatrixArray<double>& Hs = *_hessian.Hs[t1][t2];

                     for (int k = 0; k < nMeasurements; ++k)
                     {
                        int const ix1 = costFun._correspondingParams[k][i1];
                        int const id1 = this->getParamId(t1, ix1);
                        int const ix2 = costFun._correspondingParams[k][i2];
                        int const id2 = this->getParamId(t2, ix2);

#if 1 || defined(ONLY_UPPER_TRIANGULAR_HESSIAN)
                        if (id1 > id2) continue; // only store the upper diagonal blocks
#endif
                        int const n = index[k][i1][i2];
                        assert(n < Hs.count());

                        double const c1 = Cs[k][0], c2 = Cs[k][1], c3 = Cs[k][2];

                        //makeZeroMatrix(C);
                        double const factor = c1+c2+c3;
                        makeOuterProductMatrix(residuals._residuals[k], C); scaleMatrixIP(factor, C);
                        double const w = weights[k];
                        for (int j = 0; j < C.num_rows(); ++j) C[j][j] += w;
                        scaleMatrixIP(W, C);

                        multiply_A_B(C, Js2[k], C_J2);
                        multiply_At_B(Js1[k], C_J2, J1tJ2);
                        addMatricesIP(J1tJ2, Hs[n]);
                     } // end for (k)
                  } // end for (i2)
               } // end for (i1)
            } // end for (obj)
         } // end fillHessian()

         void evalJt_e(Vector<double>& Jt_e)
         {
            makeZeroVector(Jt_e);

            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               NLSQ_Residuals const& residuals = *_residuals[obj];

               Vector<double>   const& errors  = _cached_errors[obj];
               Vector<double>   const& weights = _cached_weights[obj];
               Vector<double>   const& dws     = _cached_dws[obj];
               Vector<double>   const& bs      = _cached_bs[obj];
               double           const  W       = _weighting[obj];

               int const nParamTypes = costFun._usedParamTypes.size(), nMeasurements = costFun._nMeasurements;

               for (int i = 0; i < nParamTypes; ++i)
               {
                  int const paramType = costFun._usedParamTypes[i], paramDim = _paramDesc.dimension[paramType];

                  MatrixArray<double> const& J = *residuals._Js[i];

                  Vector<double> Jkt_e(paramDim);

                  for (int k = 0; k < nMeasurements; ++k)
                  {
                     int const id = costFun._correspondingParams[k][i];
                     int const dstRow = _paramTypeRowStart[paramType] + id*paramDim;

                     double const dw = dws[k], b = bs[k];

                     multiply_At_v(J[k], residuals._residuals[k], Jkt_e);
                     //double const s = W * (weights[k] + dw*(0.5 + b));
                     double const s = W * weights[k];
                     scaleVectorIP(s, Jkt_e);
                     for (int l = 0; l < paramDim; ++l) Jt_e[dstRow + l] += Jkt_e[l];
                  } // end for (k)
               } // end for (i)
            } // end for (obj)
         } // end evalJt_e()

         bool solve_JtJ(Vector<double> &Jt_e, Vector<double> &deltaPerm, Vector<double> &delta)
         {
            bool success_LDL = true;

            int const nCols = _JtJ_Parent.size();
            //int const nnz   = _JtJ.getNonzeroCount();
            int const lnz   = _JtJ_Lp.back();

            vector<int> Li(lnz);
            vector<double> Lx(lnz);
            vector<double> D(nCols), Y(nCols);
            vector<int> workPattern(nCols), workFlag(nCols);

            int * colStarts = (int *)_JtJ.getColumnStarts();
            int * rowIdxs   = (int *)_JtJ.getRowIndices();
            double * values = _JtJ.getValues();

            int const d = LDL_numeric(nCols, colStarts, rowIdxs, values,
                                      &_JtJ_Lp[0], &_JtJ_Parent[0], &_JtJ_Lnz[0],
                                      &Li[0], &Lx[0], &D[0],
                                      &Y[0], &workPattern[0], &workFlag[0]);

            if (d == nCols)
            {
               LDL_perm(nCols, &deltaPerm[0], &Jt_e[0], &_perm_JtJ[0]);
               LDL_lsolve(nCols, &deltaPerm[0], &_JtJ_Lp[0], &Li[0], &Lx[0]);
               LDL_dsolve(nCols, &deltaPerm[0], &D[0]);
               LDL_ltsolve(nCols, &deltaPerm[0], &_JtJ_Lp[0], &Li[0], &Lx[0]);
               LDL_permt(nCols, &delta[0], &deltaPerm[0], &_perm_JtJ[0]);
            }
            else
            {
               if (optimizerVerbosenessLevel >= 2)
               {
                  cout << "Alternate_SqrtPsi_Optimizer_Base: LDL decomposition failed with d = " << d << ". Increasing lambda." << endl;
               }
               success_LDL = false;
            }
            return success_LDL;
         } // end solve_JtJ()


         double eval_robust_objective(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _costFunctions.size();

            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto const errors = cached_errors[obj];
               auto const kernel = _kernels[obj];
               double const W    = _weighting[obj];
               for (int k = 0; k < errors.size(); ++k) cost += W * kernel.psi(errors[k]);
            }
            return cost;
         }

         double eval_robust_objective()
         {
            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

            return this->eval_robust_objective(_cached_errors);
         }

//**********************************************************************

         void minimize()
         {
            this->status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (this->_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "Alternate_SqrtPsi_Optimizer: exiting since d.o.f is zero." << endl;
               this->status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = this->_JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _costFunctions.size();

            double damping_value = this->tau;

            int LDL_failures = 0;

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            for (currentIteration = 0; currentIteration < maxIterations; ++this->currentIteration)
            {
               for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

               double const initial_cost = this->eval_robust_objective(_cached_errors);

               if (optimizerVerbosenessLevel >= 2)
                  cout << "Alternate_SqrtPsi_Optimizer: iteration: " << currentIteration << ", initial |residual|^2 = " << initial_cost << " lambda = " << damping_value << endl;

               this->fillJacobians();

               for (int obj = 0; obj < nObjs; ++obj)
               {
                  NLSQ_CostFunction     & costFun   = *_costFunctions[obj];
                  NLSQ_Residuals   const& residuals = *_residuals[obj];
                  Vector<double>   const& errors    = _cached_errors[obj];
                  Vector<double>        & weights   = _cached_weights[obj];
                  Vector<double>        & dws       = _cached_dws[obj];
                  Vector<Vector3d>      & Cs        = _cached_Cs[obj];
                  Vector<double>        & bs        = _cached_bs[obj];
                  auto const            kernel      = _kernels[obj];

                  double const d2omega0 = kernel.d2weight_dr2_at0();

                  int const K = costFun._nMeasurements;
                  for (int k = 0; k < K; ++k)
                  {
                     double const r2 = errors[k], norm_r = sqrt(r2);
                     double const w = kernel.weight_fun(r2), dw = kernel.weight_fun_deriv(r2);

                     double const c1 = (norm_r > 1e-4) ? dw/norm_r : d2omega0;
                     double const c2 = (w > 1e-4) ? 0.25*sqr(dw)/w : 0.0;

                     double const denom = kernel.psi(r2) - 0.5*w*r2;
                     double const c3 = (denom > 1e-4) ? 0.125*sqr(dw)*r2/denom : 2.0;

                     weights[k] = w; dws[k] = dw;

                     Cs[k][0] = c1; Cs[k][1] = c2; Cs[k][2] = c3;
                     bs[k]    = 0;

                     double const lam = w + r2 * (c1 + c2 + c3);
                     if (0 || lam < 0)
                     {
                        //cout << " lambda(H) = " << lam << "  c = (" << c1 << "," << c2 << "," << c3 << ")  norm_r = " << norm_r << " w = " << w << " dw = " << dw << endl;
                        Cs[k][0] += std::max(0.0, w/r2 - (c1 + c2 + c3)) + 1e-6;
                        //Cs[k][0] = Cs[k][1] = Cs[k][2] = 0.0;

                        double const lam2 = w + r2 * (Cs[k][0] + Cs[k][1] + Cs[k][2]);
                        if (lam2 < 0)
                           cout << "lambda2(H) = " << lam2 << " lambda(H) = " << lam << "  c = (" << c1 << "," << c2 << "," << c3 << ")  norm_r = " << norm_r << " w = " << w << " dw = " << dw << endl;
                     }
                  } // end for (k)
               } // end for (obj)

               this->evalJt_e(Jt_e); // Jt_e holds omega(e) J^T e at this point

               double const norm_Linf_Jt_e = norm_Linf(Jt_e);

               if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
               {
                  if (optimizerVerbosenessLevel >= 2) cout << "Alternate_SqrtPsi_Optimizer: converged due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
                  break;
               }

               scaleVectorIP(-1.0, Jt_e);

               this->fillHessian();
               bool success_LDL = false;

               // Augment the diagonals
               for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
               {
                  MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
                  vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
                  int const dim = Hs.num_cols(), count = Hs.count();

                  // Only augment those with the same parameter id
                  for (int n = 0; n < count; ++n)
                  {
                     if (nzPairs[n].first != nzPairs[n].second) continue;
                     for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
                  }
               } // end for (paramType)

               this->fillJtJ();
               success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
               if (!success_LDL) ++LDL_failures;

               this->copyToAllParameters(&x_saved[0]);

               bool success_decrease = false;
               if (success_LDL)
               {
                  double const deltaSqrLength = sqrNorm_L2(delta);
                  double const paramLength = this->getParameterLength();

                  if (optimizerVerbosenessLevel >= 3)
                     cout << "Alternate_SqrtPsi_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                  if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "Alternate_SqrtPsi_Optimizer: converged due to small update, deltaSqrLength = " << deltaSqrLength << endl;
                     break;
                  }

                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     int const paramDim = _paramDesc.dimension[paramType];
                     int const count    = _paramDesc.count[paramType];
                     int const rowStart = _paramTypeRowStart[paramType];

                     VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                     this->updateParameters(paramType, deltaParam);
                  } // end for (paramType)
                  this->finishUpdateParameters();

                  // Check if new cost is better than best one at current level
                  for (int obj = 0; obj < nObjs; ++obj) this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

                  double const current_cost  = this->eval_robust_objective(_cached_errors);

                  success_decrease = (current_cost < initial_cost);
                  if ((optimizerVerbosenessLevel >= 1 && success_decrease) || (optimizerVerbosenessLevel >= 2))
                     cout << "Alternate_SqrtPsi_Optimizer: iteration: " << setw(3) << currentIteration << " previous cost = " << setw(12) << initial_cost
                          << " new cost = " << setw(12) << current_cost << " lambda = " << damping_value << endl;
               } // end if (success_LDL)

               if (success_decrease)
               {
                  damping_value = std::max(damping_value_min, damping_value / 10);
               }
               else
               {
                  if (success_LDL)
                  {
                     this->copyFromAllParameters(&x_saved[0]);
                  }
                  damping_value *= 10;
               }
            } // end for (currentIteration)

            if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
            {
               cout << "Alternate_SqrtPsi_Optimizer: reached maximum number of iterations, exiting." << endl;
            }

            if (optimizerVerbosenessLevel >= 2)
               cout << "Leaving Alternate_SqrtPsi_Optimizer::minimize(): LDL_failures = " << LDL_failures << endl;
         } // end minimize()
   }; // end struct Alternate_SqrtPsi_Optimizer

} // end namespace Robust_LSQ

#endif
