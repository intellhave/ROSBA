#ifndef ROBUST_LSQ_SCHUR_PCG_LIFTED_H
#define ROBUST_LSQ_SCHUR_PCG_LIFTED_H

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


    template <typename Lifted_Kernel, bool use_Gauss_Newton = false, typename Parametrization = Default_Paramtrization>

        struct Robust_LSQ_Optimizer_Schur_PCG_Lifted : public Robust_LSQ_Optimizer_Schur_PCG_Base {

            Robust_LSQ_Optimizer_Schur_PCG_Lifted(NLSQ_ParamDesc const &paramDesc, 
                    std::vector<NLSQ_CostFunction *> const &costFunctions,
                    std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions,
                    std::vector<Lifted_Kernel> const &kernels, 
                    std::vector<double> const &weighting):
                Robust_LSQ_Optimizer_Schur_PCG_Base(paramDesc, costFunctions, robustCostFunctions),

              _kernels(kernels), _weighting(weighting),
              _u_params(costFunctions.size()), _u_saved(costFunctions.size()),
              _cached_errors(costFunctions.size()), _cached_weights(costFunctions.size()),
              _cached_alphas(costFunctions.size()), _cached_betas(costFunctions.size()), _cached_deltas(costFunctions.size())
            {

                int const nObjs = _costFunctions.size();
                for (int obj = 0; obj < nObjs; ++obj) 
                    _u_params[obj].newsize(_costFunctions[obj]->_nMeasurements);
                for (int obj = 0; obj < nObjs; ++obj) 
                    _u_saved[obj].newsize(_costFunctions[obj]->_nMeasurements);
                for (int obj = 0; obj < nObjs; ++obj) 
                    _cached_errors[obj].newsize(_costFunctions[obj]->_nMeasurements);
                for (int obj = 0; obj < nObjs; ++obj) 
                    _cached_weights[obj].newsize(_costFunctions[obj]->_nMeasurements);
                for (int obj = 0; obj < nObjs; ++obj) 
                    _cached_alphas[obj].newsize(_costFunctions[obj]->_nMeasurements);
                for (int obj = 0; obj < nObjs; ++obj) 
                    _cached_betas[obj].newsize(_costFunctions[obj]->_nMeasurements);
                for (int obj = 0; obj < nObjs; ++obj) 
                    _cached_deltas[obj].newsize(_costFunctions[obj]->_nMeasurements);

                for (int obj = 0; obj < nObjs; ++obj) fillVector(Parametrization::u(initial_lifted_weight), this->_u_params[obj]);
            }

            vector<Lifted_Kernel> const _kernels;
            vector<double>        const _weighting;
            vector<Vector<double> > _u_params, _u_saved, _cached_errors, _cached_weights;
            vector<Vector<double> > _cached_alphas, _cached_betas, _cached_deltas;

            virtual void copyToAllParameters(double * dst) = 0;
            virtual void copyFromAllParameters(double const * src) = 0;

            void cache_errors(NLSQ_CostFunction& costFun, NLSQ_Residuals &residuals, 
                    Vector<double> &errors) const
            {
                costFun.preIterationCallback();
                costFun.initializeResiduals();
                costFun.evalAllResiduals(residuals._residuals);

                for (int k = 0; k < costFun._nMeasurements; ++k) 
                    errors[k] = sqrNorm_L2(residuals._residuals[k]);
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
                    auto const& u = _u_params[obj];
                    for (int k = 0; k < errors.size(); ++k)
                    {
                        double const w = Parametrization::w(u[k]);
                        cost1 += W * 0.5*w*errors[k];
                        cost2 += W * kernel.gamma(w);
                    }
                }
                //cout << "cost1 = " << cost1 << " cost2 = " << cost2 << endl;
                return cost1 + cost2;
            }

            void evalJt_e() 
            {
                Vector<double> Jt_eA(_dimensionA), Jt_eB(_dimensionB);
                for (int i = 0; i < _countA; i++) makeZeroVector(_Jt_eA[i]);
                for (int j = 0; j < _countB; j++) makeZeroVector(_Jt_eB[j]);

                for (int obj = 0; obj < _costFunctions.size(); obj++){

                    Vector<double> const& weights = _cached_weights[obj];
                    Vector<double> const& alphas  = _cached_alphas[obj];
                    Vector<double> const& betas   = _cached_betas[obj];
                    Vector<double> const& deltas  = _cached_deltas[obj];
                    double         const  W       = _weighting[obj];
                    MatrixArray<double> const &Js1 = *_residuals[obj]->_Js[0];
                    MatrixArray<double> const &Js2 = *_residuals[obj]->_Js[1];
                    auto const &residuals = _residuals[obj]->_residuals;
                    auto const &costFun = *_costFunctions[obj];


                    for (int k = 0; k < _nMeasurements; k++){
                        int const i = costFun._correspondingParams[k][0];
                        int const j = costFun._correspondingParams[k][1];

                        multiply_At_v(Js1[k], residuals[k], Jt_eA);
                        multiply_At_v(Js2[k], residuals[k], Jt_eB);

                        if (1 || !use_weights_EPI)
                        {
                            double const s = W * (weights[k] - alphas[k]*deltas[k]/betas[k]);
                            scaleVectorIP(s, Jt_eA);
                            scaleVectorIP(s, Jt_eB);
                        }
                        else
                        {
                            double const s = W * weights[k];
                            scaleVectorIP(s, Jt_eA);
                            scaleVectorIP(s, Jt_eB);
                        } // end if

                        addVectorsIP(Jt_eA, _Jt_eA[i]);
                        addVectorsIP(Jt_eB, _Jt_eB[j]);
                    }
                }
            }


            void evalJt_eNew() 
            {
                for (int i = 0; i < _countA; i++) makeZeroVector(_Jt_eA[i]);
                for (int j = 0; j < _countB; j++) makeZeroVector(_Jt_eB[j]);

                for (int obj = 0; obj < _costFunctions.size(); obj++){

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

                            int const cam = costFun._correspondingParams[k][0];
                            int const point = costFun._correspondingParams[k][1];
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
                            /* for (int l = 0; l < paramDim; ++l) Jt_e[dstRow + l] += Jkt_e[l]; */

                            if (i == 0)
                                addVectorsIP(Jkt_e, _Jt_eA[cam]);
                            else
                                addVectorsIP(Jkt_e, _Jt_eB[point]);
                            
                        } // end for (k)
                    } // end for (i)
                }//end for obj
            }

            void fillHessian(){

                Matrix<double> J1tJ1(_dimensionA, _dimensionA), 
                    J2tJ2(_dimensionB, _dimensionB), J1tJ2(_dimensionA, _dimensionB);

                for (int i = 0; i < _countA; ++i) makeZeroMatrix(_Us[i]);
                for (int j = 0; j < _countB; ++j) makeZeroMatrix(_Vs[j]);
                for (int k = 0; k < _nMeasurements; ++k) makeZeroMatrix(_Ws[k]);
                int const nObjs = _costFunctions.size();

                for (int obj = 0; obj < nObjs; ++obj)
                {

                    NLSQ_CostFunction& costFun = *_costFunctions[obj];
                    MatrixArray<double> const &Js1 = *_residuals[obj]->_Js[0];
                    MatrixArray<double> const &Js2 = *_residuals[obj]->_Js[1];

                    int const nParamTypes = costFun._usedParamTypes.size();
                    int const nMeasurements = costFun._nMeasurements, 
                        residualDim = costFun._measurementDimension;

                    Matrix<double> C(residualDim, residualDim);
                    Matrix<double> C_J1 (residualDim, _dimensionA);
                    Matrix<double> C_J2 (residualDim, _dimensionB);

                    MatrixArray<int> const& index = *_hessianIndices[obj];
                    NLSQ_Residuals   const& residuals = *_residuals[obj];

                    Vector<double> const& errors  = _cached_errors[obj];
                    Vector<double> const& weights = _cached_weights[obj];
                    Vector<double> const& alphas  = _cached_alphas[obj];
                    Vector<double> const& betas   = _cached_betas[obj];
                    Vector<double> const& deltas  = _cached_deltas[obj];
                    double         const  W       = _weighting[obj];

                    for (int k = 0; k < _nMeasurements; k++){

                        makeOuterProductMatrix(residuals._residuals[k], C);
                        if (1 || !use_weights_EPI)
                            scaleMatrixIP(-sqr(alphas[k])/betas[k], C);
                        else
                            scaleMatrixIP(-alphas[k], C);

                        double const w = weights[k];
                        for (int j = 0; j < C.num_rows(); ++j) C[j][j] += w;
                        scaleMatrixIP(W, C);

                        int const cam = costFun._correspondingParams[k][0];
                        int const point = costFun._correspondingParams[k][1];

                        multiply_A_B(C, Js1[k], C_J1);
                        multiply_A_B(C, Js2[k], C_J2);

                        //Camera JtJ
                        multiply_At_B(Js1[k], C_J1,  J1tJ1); 
                        addMatricesIP(J1tJ1, _Us[cam]);

                        //Point JtJ
                        multiply_At_B(Js2[k], C_J2, J2tJ2); 
                        addMatricesIP(J2tJ2, _Vs[point]);

                        //Camera vs Point JtJ
                        multiply_At_B(Js1[k], C_J2 , J1tJ2); 
                        addMatricesIP(J1tJ2, _Ws[k]);
                    }
                }//end for obj
            }


            void fillHessianNew(){


                for (int i = 0; i < _countA; ++i) makeZeroMatrix(_Us[i]);
                for (int j = 0; j < _countB; ++j) makeZeroMatrix(_Vs[j]);
                for (int k = 0; k < _nMeasurements; ++k) makeZeroMatrix(_Ws[k]);
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


                            for (int k = 0; k < nMeasurements; ++k)
                            {
                                int const cam = costFun._correspondingParams[k][0];
                                int const point = costFun._correspondingParams[k][1];

                                int const ix1 = costFun._correspondingParams[k][i1];
                                int const id1 = this->getParamId(t1, ix1);
                                int const ix2 = costFun._correspondingParams[k][i2];
                                int const id2 = this->getParamId(t2, ix2);

                                if (id1 > id2) continue; // only store the upper diagonal blocks

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
                                if (i1==0 && i2 == 0)
                                    addMatricesIP(J1tJ2, _Us[cam]);
                                else if(i1==1 && i2==1)
                                    addMatricesIP(J1tJ2, _Vs[point]);
                                else {
                                    addMatricesIP(J1tJ2, _Ws[k]);
                                }
                            } // end for (k)
                        } // end for (i2)
                    } // end for (i1)
                }//end for obj
            }

            void minimize()
            {
                std::default_random_engine rng;
                std::normal_distribution<double> dist(0.0, 1e-6);
                this->status = LEVENBERG_OPTIMIZER_TIMEOUT;

                int const totalParamDimension = _totalParamDimension;

                int const nObjs = _costFunctions.size();
                double &damping_value = _damping;
                damping_value = this->tau;
                int LDL_failures = 0;
                for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);
                _timer.start();
                currentIteration = 0;

                /* for (currentIteration = 0; currentIteration < maxIterations; */ 
                /*         ++this->currentIteration) */
                while (currentIteration < maxIterations)
                {
                    ++this->currentIteration;
                    for (int obj = 0; obj < nObjs; ++obj) 
                        this->cache_errors(*_costFunctions[obj], *_residuals[obj], _cached_errors[obj]);

                    double const initial_cost = use_lifted_varpro ? this->eval_robust_objective(_cached_errors) : this->eval_lifted_cost(_cached_errors);
                    double init_robust_cost = this->eval_robust_objective(_cached_errors);

                    cout << "Lifted_Optimizer: iteration: " << currentIteration 
                        << ", initial |residual|^2 = " << initial_cost 
                        << " robust cost = " << init_robust_cost 
                        << " lambda = " << damping_value << endl;                  

                    this->updateStat(init_robust_cost);
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

                        double const d2gamma1 = kernel.d2gamma_dw2(1.0), 
                               d3gamma1 = kernel.d3gamma_dw3(1.0);

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
                                hessian += r2;
                                double const denom1 = kernel.gamma(w);
                                if (fabs(denom1) > 1e-6)
                                    hessian += 0.5*w*sqr(dgamma_dw)/denom1;
                                else
                                    hessian += 0.5*w*(2*d2gamma1 + (w-1.0)*1.3333333*d3gamma1);

                                betas[k] = hessian + 1e-6;
                            }
                            else
                            {
                                double const u = us[k];
                                Vector3d const vals = Parametrization::w_dw_d2w(u);
                                double const w = vals[0], dw = vals[1], 
                                       d2w = vals[2], d2w0 = Parametrization::d2w0();
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

                    this->evalJt_eNew(); // Jt_e holds omega(e) J^T e at this point
                    this->fillHessianNew();
                    this->addDamping();
                    this->prepare_schur();
                    this->fillJt_e_schur();
                    this->fill_M();

                    this->solveJtJ();
                    //Update parameters
                    this->saveAllParameters();
                    for (int obj = 0; obj < nObjs; ++obj) 
                        _u_saved[obj] = _u_params[obj];

                    this->updateParameters(0, _deltaA);
                    this->updateParameters(1, _deltaB);
                    this->finishUpdateParameters();

                    bool success_decrease = false;
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

                                double * deltaParam;
                                if (i ==  0) deltaParam = &_DeltaA[0];
                                else deltaParam = &_DeltaB[0];

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


                            for (int k = 0; k < K; ++k)
                            {
                                double const du = -(deltas[k] + alphas[k]*delta_Jt_e[k]) / betas[k];
                                u[k] += du;
                                //if (fabs(u[k]) > 1.0) u[k] = (u[k] > 0) ? 1.0 : -1.0;
                                norm_du += sqr(du) / K;
                            } // end for (k)
                            /* cout << "u(after) = "; displayVector_n(u, 20); */
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
                        /* cout << "Lifted_Optimizer: iteration: " << setw(3) << currentIteration << " previous lifted cost = " << setw(12) << initial_cost */
                        /*     << " new lifted cost = " << setw(12) << current_cost << " actual robust cost = " << setw(12) << actual_cost << " lambda = " << damping_value << endl; */

                    }

                    /* if (optimizerVerbosenessLevel >= 2 && !success_decrease) cout << "Lifted_Optimizer: iteration: " << currentIteration << " previous lifted cost = " << initial_cost */
                    /*     << " new lifted cost = " << current_cost << " actual robust cost = " << actual_cost << " lambda = " << damping_value << endl; */


                    if (success_decrease)
                    {
                        damping_value = std::max(damping_value_min, damping_value / 10);
                    }
                    else
                    {
                        this->restoreAllParameters();
                        for (int obj = 0; obj < nObjs; ++obj) 
                            _u_params[obj] = _u_saved[obj];
                        damping_value *= 10;
                        this->currentIteration--;
                    }

                } // for current iteration

            }//end minimize
        }; //end struct definition

};//End namespace Robust_LSQ
#endif
