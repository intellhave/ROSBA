// -*- C++ -*-
#ifndef UTILITIES_COMMON_H
#define UTILITIES_COMMON_H

#include "robust_lsq_common.h"
#include "Math/v3d_linear.h"
#include "Math/v3d_linear_tnt.h"
#include "Geometry/v3d_mviewutilities.h"

namespace V3D
{
   using namespace std;

   inline double
   norm_Linf(VectorArray<double> const& vs)
   {
      double res = 0;
      for (int i = 0; i < vs.count(); ++i) res = std::max(res, norm_Linf(vs[i]));
      return res;
   }

   inline double
   sqrNorm_L2(VectorArray<double> const& vs)
   {
      double res = 0;
      for (int i = 0; i < vs.count(); ++i) res += sqrNorm_L2(vs[i]);
      return res;
   }

   inline double norm_L2(VectorArray<double> const& vs) { return sqrt(sqrNorm_L2(vs)); }

   inline double
   inner_product(VectorArray<double> const& vs, VectorArray<double> const& ws)
   {
      double res = 0;
      for (int i = 0; i < vs.count(); ++i) res += innerProduct(vs[i], ws[i]);
      return res;
   }

   inline double
   inner_product(VectorArrayAdapter<double> const& vs, VectorArray<double> const& ws)
   {
      double res = 0;
      for (int i = 0; i < vs.count(); ++i) res += innerProduct(vs[i], ws[i]);
      return res;
   }

   inline void
   copyVectorArray(VectorArray<double> const& src, VectorArray<double> &dst)
   {
      for (int i = 0; i < src.count(); ++i) copyVector(src[i], dst[i]);
   }

   template <typename Vec_Array>
   inline void
   add_scaled_vector_array_IP(double const alpha, VectorArray<double> const& v, Vec_Array &dst)
   {
      int const d = dst.size();
      for (int i = 0; i < dst.count(); ++i)
         for (int j = 0; j < d; ++j) dst[i][j] += alpha * v[i][j];
   }

//**********************************************************************

   template <int whichParam>
   struct Measurement_Visibility
   {
         Measurement_Visibility(int const count, int const nMeasurements, Matrix<int> const& correspondingParams)
            : _measurementStarts(count), _measurementCounts(count), _measurementIxs(nMeasurements)
         {
            // Determine the mapping from B variables (e.g. points) to their measurement indices
            fillVector(0, _measurementCounts);
            for (int k = 0; k < nMeasurements; ++k)
            {
               int const j = correspondingParams[k][whichParam];
               ++_measurementCounts[j];
            }

            _measurementStarts[0] = 0;
            for (int j = 1; j < count; ++j) _measurementStarts[j] = _measurementStarts[j-1] + _measurementCounts[j-1];
            fillVector(0, _measurementCounts);

            for (int k = 0; k < nMeasurements; ++k)
            {
               int const j = correspondingParams[k][whichParam];
               int const pos = _measurementStarts[j] + _measurementCounts[j];
               _measurementIxs[pos] = k;
               ++_measurementCounts[j];
            }
         } // end Measurement_Visibility()

         Measurement_Visibility(int const count, vector<PointMeasurement> const& ms)
            : _measurementStarts(count), _measurementCounts(count), _measurementIxs(ms.size())
         {
            // Determine the mapping from B variables (e.g. points) to their measurement indices

            int const nMeasurements = ms.size();

            fillVector(0, _measurementCounts);
            for (int k = 0; k < nMeasurements; ++k)
            {
               int const j = (whichParam == 1) ? ms[k].id : ms[k].view;
               ++_measurementCounts[j];
            }

            _measurementStarts[0] = 0;
            for (int j = 1; j < count; ++j) _measurementStarts[j] = _measurementStarts[j-1] + _measurementCounts[j-1];
            fillVector(0, _measurementCounts);

            for (int k = 0; k < nMeasurements; ++k)
            {
               int const j = (whichParam == 1) ? ms[k].id : ms[k].view;
               int const pos = _measurementStarts[j] + _measurementCounts[j];
               _measurementIxs[pos] = k;
               ++_measurementCounts[j];
            }
         } // end Measurement_Visibility()

         int paramCount() const { return _measurementStarts.size(); }

         vector<int> _measurementStarts, _measurementCounts, _measurementIxs;
   }; // end struct Measurement_Visibility

//**********************************************************************

   /* inline Matrix3x3d */
   /* project_to_SO3(Matrix3x3d const R) */
   /* { */
   /*    Matrix<double> RR(3, 3); */
   /*    copyMatrix(R, RR); */
   /*    SVD<double> svd(RR); */
   /*    Matrix3x3d res; */
   /*    multiply_A_Bt(svd.getU(), svd.getV(), res); */
   /*    return res; */
   /* } */

   /* struct SimpleDistortionFunction */
   /* { */
   /*       double k1, k2; */

   /*       SimpleDistortionFunction() */
   /*          : k1(0), k2(0) */
   /*       { } */

   /*       Vector2d operator()(Vector2d const& xu) const */
   /*       { */
   /*          double const r2 = xu[0]*xu[0] + xu[1]*xu[1]; */
   /*          double const r4 = r2*r2; */
   /*          double const kr = 1 + k1*r2 + k2*r4; */

   /*          Vector2d xd; */
   /*          xd[0] = kr * xu[0]; */
   /*          xd[1] = kr * xu[1]; */
   /*          return xd; */
   /*       } */

   /*       Matrix2x2d derivativeWrtRadialParameters(Vector2d const& xu) const */
   /*       { */
   /*          double const r2 = xu[0]*xu[0] + xu[1]*xu[1]; */
   /*          double const r4 = r2*r2; */

   /*          Matrix2x2d deriv; */

   /*          deriv[0][0] = xu[0] * r2; // d xd/d k1 */
   /*          deriv[0][1] = xu[0] * r4; // d xd/d k2 */
   /*          deriv[1][0] = xu[1] * r2; // d yd/d k1 */
   /*          deriv[1][1] = xu[1] * r4; // d yd/d k2 */
   /*          return deriv; */
   /*       } */

   /*       Matrix2x2d derivativeWrtUndistortedPoint(Vector2d const& xu) const */
   /*       { */
   /*          double const r2 = xu[0]*xu[0] + xu[1]*xu[1]; */
   /*          double const r4 = r2*r2; */
   /*          double const kr = 1 + k1*r2 + k2*r4; */
   /*          double const dkr = 2*k1 + 4*k2*r2; */

   /*          Matrix2x2d deriv; */
   /*          deriv[0][0] = kr + xu[0] * xu[0] * dkr; // d xd/d xu */
   /*          deriv[0][1] =      xu[0] * xu[1] * dkr; // d xd/d yu */
   /*          deriv[1][0] = deriv[0][1];              // d yd/d xu */
   /*          deriv[1][1] = kr + xu[1] * xu[1] * dkr; // d yd/d yu */
   /*          return deriv; */
   /*       } */
   /* }; // end struct SimpleDistortionFunction */

//**********************************************************************

   /* inline double sqr(double const x) { return x*x; } */
   /* inline double sgn(double const x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); } */

//**********************************************************************

   /* inline void */
   /* triangulateLinear(vector<CameraMatrix> const& cams, vector<SimpleDistortionFunction> const& distortions, */
   /*                   vector<Vector2d> const& measurements, Measurement_Visibility<1> const& vis_info, vector<int> const& correspondingView, */
   /*                   int const startB, int const endB, vector<Vector3d>& Xs) */
   /* { */
   /*    for (int j = startB; j < endB; ++j) */
   /*    { */
   /*       int const count = vis_info._measurementCounts[j], start = vis_info._measurementStarts[j]; */
   /*       int const nRows = 2*count, nCols = 3; */

   /*       Matrix<double> A(nRows, nCols); */
   /*       Vector<double> b(nRows); */

   /*       for (int p = 0; p < count; ++p) */
   /*       { */
   /*          int const k = vis_info._measurementIxs[start + p]; */
   /*          int const i = correspondingView[k]; */

   /*          double const x = measurements[k][0], y = measurements[k][1]; */

   /*          Matrix3x4d const& P = cams[i].getProjection(); */

   /*          A[2*p+0][0] = x*P[2][0] - P[0][0]; A[2*p+0][1] = x*P[2][1] - P[0][1]; A[2*p+0][2] = x*P[2][2] - P[0][2]; */
   /*          A[2*p+1][0] = y*P[2][0] - P[1][0]; A[2*p+1][1] = y*P[2][1] - P[1][1]; A[2*p+1][2] = y*P[2][2] - P[1][2]; */

   /*          b[2*p+0] = P[0][3] - x*P[2][3]; b[2*p+1] = P[1][3] - y*P[2][3]; */
   /*       } // end for (p) */

   /*       QR<double> qr(A); */
   /*       Vector<double> const X = qr.solve(b); */
   /*       copyVector(X, Xs[j]); */
   /*    } // end for (j) */
   /* } // end triangulateLinear() */

} // end namespace V3D

#endif
