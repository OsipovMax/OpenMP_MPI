#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <time.h>

using namespace std;

// crs format
struct crsMatrix {
  int N;  // matrix size

  vector<double> value;
  vector<int> col;
  vector<int> indexRow;
};

// generation portrait of the matrix analysis area
crsMatrix GenerateMatrix1(int x, int y, int z) {
  int i_str = 0, _size, t;
  double tv = 0.0, temp;
  crsMatrix _crsMtr;
  _size = x * y * z;
  _crsMtr.N = _size;
  _crsMtr.indexRow.push_back(0);
  int cm = 0, pip = 0, m = -1;
  bool flag = true;
  for (int k = 0; k <= z - 1; ++k) {
    for (int j = 0; j <= y - 1; ++j) {
      for (int i = 0; i <= x - 1; ++i) {
        i_str = k * (x * y) + j * x + i;
        m++;
        if (i_str == m) {
          if (k > 0) {
            temp = sin(i_str + i_str - x * y + 1);
            _crsMtr.col.push_back(i_str - x * y);
            _crsMtr.value.push_back(temp);
            tv += fabs(temp);
          }
          if (j > 0) {
            temp = sin(i_str + i_str - x + 1);
            _crsMtr.col.push_back(i_str - x);
            _crsMtr.value.push_back(temp);
            tv += fabs(temp);
          }
          if (i > 0) {
            temp = sin(i_str + i_str - 1 + 1);
            _crsMtr.col.push_back(i_str - 1);
            _crsMtr.value.push_back(temp);
            tv += fabs(temp);
          }
          if (1) {
            _crsMtr.col.push_back(m);
            _crsMtr.value.push_back(-1);
            t = _crsMtr.value.size() - 1;
          }
          if (i < x - 1) {
            temp = sin(i_str + i_str + 1 + 1);
            _crsMtr.col.push_back(i_str + 1);
            _crsMtr.value.push_back(temp);
            tv += fabs(temp);
          }
          if (j < y - 1) {
            temp = sin(i_str + i_str + x + 1);
            _crsMtr.col.push_back(i_str + x);
            _crsMtr.value.push_back(temp);
            tv += fabs(temp);
          }
          if (k < z - 1) {
            temp = sin(i_str + i_str + x * y + 1);
            _crsMtr.col.push_back(i_str + x * y);
            _crsMtr.value.push_back(temp);
            tv += fabs(temp);
          }
          flag = false;
          _crsMtr.value[t] = 1.1 * tv;
          tv = 0.0;
          _crsMtr.indexRow.push_back(_crsMtr.col.size());
        }
      }
    }
  }
  return _crsMtr;
}

// sequential implementation of a scalar product
double dot(vector<double> &v1, vector<double> &v2) {
  double result = 0.0;
  for (int i = 0; i < v1.size(); ++i) {
    result += v1[i] * v2[i];
  }
  return result;
}
// sequential implementation of linear combination of vectors
void axpby(vector<double> &v1, vector<double> v2, double a, double b) {
  for (int i = 0; i < v1.size(); i++) {
    v1[i] = v1[i] * a + v2[i] * b;
  }
}
// sequential realization of matrix-vector product
void SpMV(crsMatrix &m1, vector<double> v1, vector<double> &v2) {
  for (int i = 0; i < v2.size(); i++) v2[i] = 0.0;

  for (int i = 0; i < m1.N; ++i) {
    int j1 = m1.indexRow[i];
    int j2 = m1.indexRow[i + 1];
    for (int j = j1; j < j2; ++j) {
      v2[i] += m1.value[j] * v1[m1.col[j]];
    }
  }
}
//------------------------------------------------------------------------------------
// optimized operation based on the unrolled cycle
double Optdot(vector<double> &v1, vector<double> &v2) {
  double result1 = 0.0, result2 = 0.0, result3 = 0.0, result4 = 0.0,
         result5 = 0.0;
  int N = v1.size();
  const int N3 = (N / 3) * 3;
  for (int i = 0; i < N3; i += 3) {
    result1 += v1[i] * v2[i];
    result2 += v1[i + 1] * v2[i + 1];
    result3 += v1[i + 2] * v2[i + 2];
  }
  for (int i = N3; i < N; i++) result1 += v1[i] * v2[i];

  return (result3 + result2 + result1);
}

void Optaxpby(vector<double> &v1, vector<double> v2, double a, double b) {
  double result1 = 0.0, result2 = 0.0, result3 = 0.0, result4 = 0.0,
         result5 = 0.0;
  int N = v1.size();
  int N3 = (N / 3) * 3;
  for (int i = 0; i < N3; i += 3) {
    v1[i] = v1[i] * a + v2[i] * b;
    v1[i + 1] = v1[i + 1] * a + v2[i + 1] * b;
    v1[i + 2] = v1[i + 2] * a + v2[i + 2] * b;
  }
  for (int i = N3; i < N; i++) v1[i] = v1[i] * a + v2[i] * b;
}

void OptSpMV(crsMatrix &m1, vector<double> v1, vector<double> &v2) {
  for (int i = 0; i < v2.size(); i++) v2[i] = 0.0;
  for (int i = 0; i < m1.N; ++i) {
    int j1 = m1.indexRow[i];
    int j2 = m1.indexRow[i + 1];
    const int N2 = (j2 / 2) * 2;
    if (N2 != j2) {
      for (int j = j1; j < N2; j += 2) {
        v2[i] += m1.value[j] * v1[m1.col[j]];
        v2[i] += m1.value[j + 1] * v1[m1.col[j + 1]];
      }
      for (int p = (j2 - ((j2 - j1) % 2)); p < j2; p++)
        v2[i] += m1.value[p] * v1[m1.col[p]];
    } else {
      for (int j = j1; j < N2 - 1; j += 2) {
        v2[i] += m1.value[j] * v1[m1.col[j]];
        v2[i] += m1.value[j + 1] * v1[m1.col[j + 1]];
      }
      for (int p = (j2 - ((j2 - j1) % 2)); p < j2; p++)
        v2[i] += m1.value[p] * v1[m1.col[p]];
    }
  }
}
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
// parallel multithreaded implementation based on OpenMP directives
double parallelDot(vector<double> &v1, vector<double> &v2, int num_th) {
  double result = 0.0, temp = 0.0;
  omp_set_dynamic(0);
  omp_set_num_threads(num_th);

#pragma omp parallel for reduction(+ : result)
  for (int i = 0; i < v1.size(); i++) {
    result += v1[i] * v2[i];
  }
  return result;
}

void parallelaxpby(vector<double> &v1, vector<double> v2, double a, double b,
                   int num_th) {
  double temp = 0.0;
  omp_set_dynamic(0);
  omp_set_num_threads(num_th);
  int i;
#pragma omp parallel for reduction(+ : temp)
  for (i = 0; i < v1.size(); i++) {
    temp = v1[i] * a + v2[i] * b;
    v1[i] = temp;
  }
}
void parallelSpMV(crsMatrix &m1, vector<double> v1, vector<double> &v2,
                  int num_th) {
  double result = 0.0;

  int j;
  omp_set_dynamic(0);
  omp_set_num_threads(num_th);

  for (int i = 0; i < v2.size(); i++) v2[i] = 0.0;

#pragma omp parallel for private(j)
  for (int i = 0; i < m1.N; ++i) {
    int j1 = m1.indexRow[i];
    int j2 = m1.indexRow[i + 1];
    for (j = j1; j < j2; ++j) {
      v2[i] += m1.value[j] * v1[m1.col[j]];
    }
  }
}
// sequential implementation of the BiCGSTAB method
int BicGSTAB(crsMatrix &_mtr, double tol, int maxit) {
  double mineps = 1e-15;
  double Rhoi_1 = 1.0, alphai = 1.0, wi = 1.0;
  double betai_1 = 1.0, Rhoi_2 = 1.0, alphai_1 = 1.0;
  double wi_1 = 1.0, RhoMin = 1e-60;
  int I, info = 1, v_size = _mtr.N;
  vector<double> XX, BB, RR(v_size, 0.0), RR2(v_size, 0.0), PP(v_size, 0.0),
      PP2(v_size, 0.0), SS(v_size, 0.0), VV(v_size, 0.0), TT(v_size, 0.0),
      SS2(v_size, 0.0);

  crsMatrix DD;
  DD.N = _mtr.N;
  DD.col.resize(0);
  for (int i = 0; i < _mtr.N; i++) {
    int j1 = _mtr.indexRow[i];
    int j2 = _mtr.indexRow[i + 1];
    for (int j = j1; j < j2; j++) {
      if (i == _mtr.col[j]) {
        DD.value.push_back(1 / _mtr.value[j]);
        DD.indexRow.push_back(DD.col.size());
        DD.col.push_back(i);
      }
    }
  }
  DD.indexRow.push_back(DD.col.size());
  for (int i = 0; i < _mtr.N; i++) {
    XX.push_back(0.0);
    BB.push_back(sin((double)i));
  }

  RR = BB;
  RR2 = BB;
  double initres = sqrt(dot(RR, RR));
  double eps = max(mineps, tol * initres);

  cout << "initres = " << initres << endl;
  cout << "eps = " << eps << endl;

  double res = initres;
  for (I = 0; I < maxit; I++) {
    if (info)
      cout << "It: " << I << " res = " << scientific << res
           << " tol = " << scientific << res / initres << endl;
    if (res < eps) break;
    if (res > initres / mineps) return -1;
    if (I == 0)
      Rhoi_1 = initres * initres;
    else
      Rhoi_1 = dot(RR2, RR);
    if (fabs(Rhoi_1) < RhoMin) return -1;

    if (I == 0)
      PP = RR;
    else {
      betai_1 = (Rhoi_1 * alphai_1) / (Rhoi_2 * wi_1);
      axpby(PP, RR, betai_1, 1.0);
      axpby(PP, VV, 1.0, -wi_1 * betai_1);
    }
    SpMV(DD, PP, PP2);
    SpMV(_mtr, PP2, VV);

    alphai = dot(RR2, VV);
    if (fabs(alphai) < RhoMin) return -3;
    alphai = Rhoi_1 / alphai;
    SS = RR;
    axpby(SS, VV, 1.0, -alphai);

    SpMV(DD, SS, SS2);
    SpMV(_mtr, SS2, TT);
    wi = dot(TT, TT);
    if (fabs(wi) < RhoMin) return -4;
    wi = dot(TT, SS) / wi;
    if (fabs(wi) < RhoMin) return -5;
    axpby(XX, PP2, 1.0, alphai);
    axpby(XX, SS2, 1.0, wi);
    RR = SS;
    axpby(RR, TT, 1.0, -wi);
    alphai_1 = alphai;
    Rhoi_2 = Rhoi_1;
    wi_1 = wi;
    res = sqrt(dot(RR, RR));
  }
  if (info) {
    cout << "Solver_BiCGSTAB: outres:" << scientific << res << endl;
    cout << "Tol: " << scientific << res / initres << endl;
  }

  return I;
}
// parallel implementation of the BiCGSTAB method
int parallel_BicGSTAB(crsMatrix &_mtr, double tol, int maxit, int nit) {
  double mineps = 1e-15;
  double Rhoi_1 = 1.0, alphai = 1.0, wi = 1.0;
  double betai_1 = 1.0, Rhoi_2 = 1.0, alphai_1 = 1.0;
  double wi_1 = 1.0, RhoMin = 1e-60;
  int I = 0, info = 1, v_size = _mtr.N;
  vector<double> XX, BB, RR(v_size, 0.0), RR2(v_size, 0.0), PP(v_size, 0.0),
      PP2(v_size, 0.0), SS(v_size, 0.0), VV(v_size, 0.0), TT(v_size, 0.0),
      SS2(v_size, 0.0);

  crsMatrix DD;
  DD.N = _mtr.N;
  DD.col.resize(0);
  for (int i = 0; i < _mtr.N; i++) {
    int j1 = _mtr.indexRow[i];
    int j2 = _mtr.indexRow[i + 1];
    for (int j = j1; j < j2; j++) {
      if (i == _mtr.col[j]) {
        DD.value.push_back(1 / _mtr.value[j]);
        DD.indexRow.push_back(DD.col.size());
        DD.col.push_back(i);
      }
    }
  }
  DD.indexRow.push_back(DD.col.size());
  for (int i = 0; i < _mtr.N; i++) {
    XX.push_back(0.0);
    BB.push_back(sin((double)i));
  }

  RR = BB;
  RR2 = BB;
  double initres = sqrt(parallelDot(RR, RR, nit));
  double eps = max(mineps, tol * initres);
  cout << "initres = " << initres << endl;
  cout << "eps = " << eps << endl;
  double res = initres;

  for (I = 0; I < maxit; I++) {
    if (info)
      cout << "It: " << I << " res = " << scientific << res
           << " tol = " << scientific << res / initres << endl;
    if (res < eps) break;
    if (res > initres / mineps) return -1;
    if (I == 0)
      Rhoi_1 = initres * initres;
    else
      Rhoi_1 = parallelDot(RR2, RR, nit);
    if (fabs(Rhoi_1) < RhoMin) return -1;
    if (I == 0)
      PP = RR;
    else {
      betai_1 = (Rhoi_1 * alphai_1) / (Rhoi_2 * wi_1);
      parallelaxpby(PP, RR, betai_1, 1.0, nit);
      parallelaxpby(PP, VV, 1.0, -wi_1 * betai_1, nit);
    }

    parallelSpMV(DD, PP, PP2, nit);
    parallelSpMV(_mtr, PP2, VV, nit);

    alphai = parallelDot(RR2, VV, nit);
    if (fabs(alphai) < RhoMin) return -3;
    alphai = Rhoi_1 / alphai;
    SS = RR;
    parallelaxpby(SS, VV, 1.0, -alphai, nit);

    parallelSpMV(DD, SS, SS2, nit);
    parallelSpMV(_mtr, SS2, TT, nit);
    wi = parallelDot(TT, TT, nit);
    if (fabs(wi) < RhoMin) return -4;
    wi = parallelDot(TT, SS, nit) / wi;
    if (fabs(wi) < RhoMin) return -5;

    parallelaxpby(XX, PP2, 1.0, alphai, nit);
    parallelaxpby(XX, SS2, 1.0, wi, nit);
    RR = SS;
    parallelaxpby(RR, TT, 1.0, -wi, nit);
    alphai_1 = alphai;
    Rhoi_2 = Rhoi_1;
    wi_1 = wi;
    res = sqrt(parallelDot(RR, RR, nit));
  }
  if (info) {
    cout << "Solver_BiCGSTAB: outres:" << scientific << res << endl;
    cout << "Tol: " << scientific << res / initres << endl;
  }
  return I;
}

int main(int argc, char *argv[]) {
  crsMatrix controlMatrix, cM1;
  vector<vector<double>> mtx;
  vector<vector<double>> tempm;
  vector<double> X;
  vector<double> Y;
  int flag = atoi(argv[7]);
  double tol_val = atof(argv[4]);
  int maxIt_val = atoi(argv[5]);
  int numthread = atoi(argv[6]);
  controlMatrix = GenerateMatrix1(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));

  double tm = omp_get_wtime();
  int r = BicGSTAB(controlMatrix, tol_val, maxIt_val);
  tm = omp_get_wtime() - tm;
  cout << "sequential_time BicGSTAB = " << tm << endl;
  cout << "It:" << r << endl;
  cout << "-------------parallel_solve_slau-------------" << endl;
  double tm1 = omp_get_wtime();
  int r1 = parallel_BicGSTAB(controlMatrix, tol_val, maxIt_val, numthread);
  tm1 = omp_get_wtime() - tm1;
  cout << "parallel_time BicGSTAB = " << tm1 << endl;
  cout << "It" << r1 << endl;

  // -------------------------------------------
  if (flag) {
    int N_cycle = 20;
    double average_1 = 0.0, average_2 = 0.0, average_3 = 0.0, average_4 = 0.0,
           average_5 = 0.0, average_6 = 0.0, average_7 = 0.0, average_8 = 0.0,
           average_9 = 0.0;
    unsigned long some_val = controlMatrix.N;
    cout << "some_val = " << some_val << endl;
    // for base operation
    const double flopaxpby = N_cycle * N_cycle * some_val * 3 * 1E-9;
    const double flop = N_cycle * N_cycle * some_val * 2 * 1E-9;
    // for solver
    const double flpaxpby = some_val * 3 * 1E-9;
    const double flp = some_val * 2 * 1E-9;

    double dot_result = 0.0, axpby_result = 0.0, SpMV_result = 0.0;
    for (int i = 0; i < some_val; i++) {
      X.push_back((double)sin(i));
      Y.push_back((double)cos(i));
    }
    vector<double> X1, X2;
    vector<double> Y1, Y2;
    X1 = X;
    X2 = X;
    Y1 = Y;
    Y2 = Y;

    // sum flop for solver
    double flp_sum = (flp * 5 + flp * 4 + flpaxpby * 6);

    cout << "sequential solver: " << (flp_sum * r) / tm << " GFLOPS" << endl;
    cout << "parallel solver: " << (flp_sum * r1) / tm1 << " GFLOPS" << endl;

    for (int i = 0; i < N_cycle; ++i) {
      double time = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j) dot_result = dot(X, Y);
      time = omp_get_wtime() - time;
      average_1 = average_1 + time;

      double time_ = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j) dot_result = Optdot(X, Y);
      time_ = omp_get_wtime() - time_;
      average_2 = average_2 + time_;

      double time1 = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j)
        dot_result = parallelDot(X, Y, numthread);
      time1 = omp_get_wtime() - time1;
      average_3 = average_3 + time1;
    }
    for (int i = 0; i < N_cycle; ++i) {
      double time = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j) axpby(X, Y, 3.0, 3.0);
      time = omp_get_wtime() - time;
      average_4 = average_4 + time;
      axpby_result = 0.0;

      double time_ = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j) axpby(X2, Y, 3.0, 3.0);
      time_ = omp_get_wtime() - time_;
      average_5 = average_5 + time_;
      axpby_result = 0.0;

      double time1 = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j)
        parallelaxpby(X1, Y, 3.0, 3.0, numthread);
      time1 = omp_get_wtime() - time1;
      average_6 = average_6 + time1;
    }
    for (int i = 0; i < N_cycle; ++i) {
      double time = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j) SpMV(controlMatrix, Y, Y);
      time = omp_get_wtime() - time;
      average_7 = average_7 + time;
      SpMV_result = 0.0;

      double time_ = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j) OptSpMV(controlMatrix, Y2, Y2);
      time_ = omp_get_wtime() - time_;
      average_8 = average_8 + time_;
      SpMV_result = 0.0;

      double time1 = omp_get_wtime();
      for (int j = 0; j < N_cycle; ++j)
        parallelSpMV(controlMatrix, Y1, Y1, numthread);
      time1 = omp_get_wtime() - time1;
      average_9 = average_9 + time1;
    }

    cout << "sequential implementation" << endl;
    cout << "--dot--";
    cout.width(16);
    cout << "time = " << average_1 << " / GFLOPS = " << flop / (average_1)
         << endl;
    cout << "--axpby--";
    cout.width(14);
    cout << "time = " << average_4 << " / GFLOPS = " << flopaxpby / (average_4)
         << endl;
    cout << "--SpMV--";
    cout.width(15);
    cout << "time = " << average_7 << " / GFLOPS = " << flop / (average_7)
         << endl;

    cout << "optimized implementation" << endl;
    cout << "--dot--";
    cout.width(16);
    cout << "time = " << average_2 << " / GFLOPS = " << flop / (average_2)
         << endl;
    cout << "--axpby--";
    cout.width(14);
    cout << "time = " << average_5 << " / GFLOPS = " << flopaxpby / (average_5)
         << endl;
    cout << "--SpMV--";
    cout.width(15);
    cout << "time = " << average_8 << " / GFLOPS = " << flop / (average_8)
         << endl;

    cout << "Parallel implementation" << endl;
    cout << "--dot--";
    cout.width(16);
    cout << "time = " << average_3 << " / Speedup = " << average_1 / average_3
         << " / GFLOPS = " << flop / (average_3) << endl;
    cout << "--axpby--";
    cout.width(14);
    cout << "time = " << average_6 << " / Speedup = " << average_4 / average_6
         << " / GFLOPS = " << flopaxpby / (average_6) << endl;
    cout << "--SpMV--";
    cout.width(15);
    cout << "time = " << average_9 << " / Speedup = " << average_7 / average_9
         << " / GFLOPS = " << flop / (average_9) << endl;
  }
}
