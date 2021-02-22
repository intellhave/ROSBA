// -*- C++ -*-
// This is a simple implementation of the LDL^T decomposition for PSD matrices

#ifndef V3D_LINEAR_LDLT_H
#define V3D_LINEAR_LDLT_H

#include "Math/v3d_linear.h"

#undef _D
#undef _L

namespace V3D
{

   template <typename Real>
   struct LDLt
   {
         LDLt(Matrix<Real> const& A)
            : _L(A.num_rows(), A.num_rows()), _D(A.num_rows()), _isPD(true)
         {
            if (A.num_rows() != A.num_cols()) throwV3DErrorHere("LDLt::LDLt(): matrix A is not square.");

            int const N = A.num_rows();

            makeIdentityMatrix(_L);

            for (int j = 0; j < N; ++j)
            {
               // First, compute the current element of D
               Real Dj = 0;
               for (int k = 0; k < j; ++k) Dj += _L[j][k]*_L[j][k]*_D[k];
               Dj = A[j][j] - Dj;
               _D[j] = Dj;

               if (Dj <= 0) { _isPD = false; return; }

               // Update L below position (j,j)
               for (int i = j+1; i < N; ++i)
               {
                  Real Lij = 0;
                  for (int k = 0; k < j; ++k)
                     Lij += _L[i][k]*_L[j][k]*_D[k];
                  Lij = A[i][j] - Lij;
                  Lij /= Dj;
                  _L[i][j] = Lij;
               } // end for (i)
            } // end for (j)
         } // end LDLt()

         Matrix<Real> const& getL() const { return _L; }
         Vector<Real> const& getD() const { return _D; }

         bool isPD() const { return _isPD; }

         template <typename Vec>
         Vector<Real> solveVec(Vec const& b) const
         {
            int const N = _L.num_rows();
            if (b.size() != N) throwV3DErrorHere("LDLt::solve(): size of vector b does not match.");

            Vector<Real> x(N);
            copyVector(b, x);

            // Solve L*y = b;
            for (int k = 0; k < N; ++k)
            {
               for (int i = 0; i < k; ++i) x[k] -= x[i]*_L[k][i];
            }

            for (int k = 0; k < N; ++k) x[k] /= _D[k];

            // Solve L'*X = Y;
            for (int k = N-1; k >= 0; --k)
            {
               for (int i = k+1; i < N; ++i) x[k] -= x[i]*_L[i][k];
            }

            return x;
         } // end solveVec()

         template <typename Vec, typename Vec2>
         bool solveVec(Vec const& b, Vec2 &x) const
         {
            if (!_isPD) return false;

            int const N = _L.num_rows();
            if (b.size() != N) throwV3DErrorHere("LDLt::solve(): size of vector b does not match.");
            if (x.size() != N) throwV3DErrorHere("LDLt::solve(): size of vector x does not match.");

            copyVector(b, x);

            // Solve L*y = b;
            for (int k = 0; k < N; ++k)
            {
               for (int i = 0; i < k; ++i) x[k] -= x[i]*_L[k][i];
            }

            for (int k = 0; k < N; ++k) x[k] /= _D[k];

            // Solve L'*X = Y;
            for (int k = N-1; k >= 0; --k)
            {
               for (int i = k+1; i < N; ++i) x[k] -= x[i]*_L[i][k];
            }

            return true;
         } // end solveVec()

         template <typename Mat>
         Matrix<Real> solveMat(Mat const& B) const
         {
            int const N = _L.num_rows();
            if (B.num_rows() != N) throwV3DErrorHere("LDLt::solve(): size of matrix B does not match.");

            int const K = B.num_cols();

            Matrix<Real> X(N, K);
            copyMatrix(B, X);

            for (int j = 0; j < K; ++j)
            {
               // Solve L*y = b;
               for (int k = 0; k < N; ++k)
               {
                  for (int i = 0; i < k; ++i) X[k][j] -= X[i][j]*_L[k][i];
               }

               for (int k = 0; k < N; ++k) X[k][j] /= _D[k];

               // Solve L'*X = Y;
               for (int k = N-1; k >= 0; --k)
               {
                  for (int i = k+1; i < N; ++i) X[k][j] -= X[i][j]*_L[i][k];
               }
            } // end for (j)

            return X;
         } // end solveMat()

         template <typename Mat>
         bool solveMat(Mat const& B, Mat &X) const
         {
            if (!_isPD) return false;

            int const N = _L.num_rows();
            if (B.num_rows() != N) throwV3DErrorHere("LDLt::solve(): size of matrix B does not match.");
            if (X.num_rows() != N) throwV3DErrorHere("LDLt::solve(): size of matrix X does not match.");

            int const K = B.num_cols();
            if (X.num_cols() != K) throwV3DErrorHere("LDLt::solve(): columns of matrices X and B do not match.");

            copyMatrix(B, X);

            for (int j = 0; j < K; ++j)
            {
               // Solve L*y = b;
               for (int k = 0; k < N; ++k)
               {
                  for (int i = 0; i < k; ++i) X[k][j] -= X[i][j]*_L[k][i];
               }

               for (int k = 0; k < N; ++k) X[k][j] /= _D[k];

               // Solve L'*X = Y;
               for (int k = N-1; k >= 0; --k)
               {
                  for (int i = k+1; i < N; ++i) X[k][j] -= X[i][j]*_L[i][k];
               }
            } // end for (j)

            return true;
         } // end solveMat()

      private:
         bool _isPD;
         Matrix<Real> _L;
         Vector<Real> _D;
   }; // end struct LDLt

} // end namespace V3D

#endif
