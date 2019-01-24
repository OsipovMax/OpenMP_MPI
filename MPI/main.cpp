#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <string>
#include <cstdlib>
#include <map>
#include <algorithm>

using namespace std;

MPI_Comm comm;
int *Part;
int numproces;
vector<int> Rows;
int rank;
vector<int> bufSend;
vector<int> bufRecv;

// lists of messages to send and receive
vector<vector<int> > rsv;
vector<vector<int> > snd;

vector<vector<int> > temp_snd;
MPI_Status status;

vector<int> nbProc;  // neighbor vector
vector<int> Halo;    // array of halo cells

struct crsMatrix {
  int N;  // matrix size
  vector<double> value;
  vector<int> col;
  vector<int> indexRow;
};
crsMatrix controlMatrix, cM1;

// vector<some_type> to buf
double *toBuf(vector<double> vec) {
  int inner_counter = 0;
  double *buf = new double[vec.size()];
  for (int i = 0; i < vec.size(); ++i) {
    buf[inner_counter] = vec[i];
    inner_counter++;
  }
  return buf;
}

// Updating the values in the halo cells
void Update(vector<double> &x) {
  double *temp_b, *transf_b_send, *transf_b_rcv, *buff_r;
  int position = 0;
  int x_size;
  char *buff;
  int pos = 0;
  MPI_Request send_req[nbProc.size()], rcv_req[nbProc.size()];
  MPI_Status send_st[nbProc.size()], rcv_st[nbProc.size()];
  MPI_Status status;
  vector<double> transferBuf;
  int nb_count = -1;
  int inn_count = -1;
  for (int i = 0; i < snd.size(); i++) {
    if (snd[i].size() > 0) {
      nb_count++;
      temp_b = new double[snd[i].size()];
      for (int j = 0; j < snd[i].size(); j++) {
        temp_b[j] = x[snd[i][j]];
      }
      MPI_Isend(temp_b, snd[i].size(), MPI_DOUBLE, nbProc[nb_count], 0, comm,
                &send_req[nb_count]);
    }
  }
  int buff_r_size;
  vector<double> temp;
  for (int i = 0; i < nbProc.size(); i++) {
    buff_r_size = rsv[nbProc[i]].size();
    buff_r = new double[buff_r_size];
    MPI_Irecv(buff_r, buff_r_size, MPI_DOUBLE, nbProc[i], 0, comm, &rcv_req[i]);
    MPI_Wait(&rcv_req[i], &rcv_st[i]);
    for (int l = 0; l < buff_r_size; ++l) {
      temp.push_back(buff_r[l]);
    }
    delete[] buff_r;
  }
  for (int g = 0; g < Halo.size(); g++) {
    x[Rows.size() + g] = temp[g];
  }
}

// "distributed" versions of basic operations
double dot(vector<double> &v1, vector<double> &v2) {
  double temp_result = 0.0;
  double result = 0.0;
  for (int i = 0; i < v1.size() - Halo.size(); ++i) {
    temp_result += v1[i] * v2[i];
  }
  MPI_Allreduce(&temp_result, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
  return result;
}

void axpby(vector<double> &v1, vector<double> v2, double a, double b) {
  for (int i = 0; i < v1.size() - Halo.size(); i++) {
    v1[i] = v1[i] * a + v2[i] * b;
  }
}

void SpMV(crsMatrix &m1, vector<double> v1, vector<double> &v2) {
  for (int i = 0; i < v2.size() - Halo.size(); i++) v2[i] = 0.0;

  for (int i = 0; i < m1.N; ++i) {
    int j1 = m1.indexRow[i];
    int j2 = m1.indexRow[i + 1];
    for (int j = j1; j < j2; ++j) {
      v2[i] += m1.value[j] * v1[m1.col[j]];
    }
  }
}

// distributed matrix generator (different processes process their part)
crsMatrix GenerateMatrix1(int x, int y, int z, int proc_axis) {
  int i_str = 0, _size, t;
  double tv = 0.0, temp;
  crsMatrix _crsMtr;
  _size = x * y * z;
  _crsMtr.N = _size;
  int cm = 0, pip = 0;
  bool flag = true;
  int proc_rank;
  int coords[3];
  int temp_coords[3], temp_rank, cnt = 0;
  Part = new int[_size];

  MPI_Comm_size(comm, &numproces);
  MPI_Comm_rank(comm, &proc_rank);
  _crsMtr.N = _size / numproces;
  Part[0] = 0;
  _crsMtr.indexRow.push_back(0);

  for (int k = 0; k < z; k++) {
    for (int j = 0; j < y; j++) {
      for (int i = 0; i < x; i++) {
        temp_coords[0] = i / (x / proc_axis);
        temp_coords[1] = j / (y / proc_axis);
        temp_coords[2] = k / (z / proc_axis);
        MPI_Cart_rank(comm, temp_coords, &temp_rank);
        Part[cnt] = temp_rank;
        cnt++;
      }
    }
  }

  int kb, ke, jb, je, ib, ie;
  MPI_Cart_coords(comm, proc_rank, 3, coords);

  ib = coords[0] * (x / proc_axis);
  ie = (coords[0] + 1) * (x / proc_axis);

  jb = coords[1] * (y / proc_axis);
  je = (coords[1] + 1) * (y / proc_axis);

  kb = coords[2] * (z / proc_axis);
  ke = (coords[2] + 1) * (z / proc_axis);
  int m = proc_rank * (x / 2) - 1;

  for (int k = kb; k < ke; ++k) {
    for (int j = jb; j < je; ++j) {
      for (int i = ib; i < ie; ++i) {
        i_str = k * (x * y) + j * x + i;
        Rows.push_back(i_str);
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
          _crsMtr.col.push_back(i_str);
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
  return _crsMtr;
}

// distributed algorithm of BicGSTAB method
int BicGSTAB(crsMatrix &_mtr, double tol, int maxit) {
  double mineps = 1e-15;
  double Rhoi_1 = 1.0, alphai = 1.0, wi = 1.0;
  double betai_1 = 1.0, Rhoi_2 = 1.0, alphai_1 = 1.0;
  double wi_1 = 1.0, RhoMin = 1e-60;
  int I, info = 1, v_size = _mtr.N + Halo.size();
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
  for (int i = 0; i < _mtr.N + Halo.size(); i++) {
    XX.push_back(0.0);
    if (i < _mtr.N) BB.push_back(cos((double)Rows[i]));
    if (i >= _mtr.N) BB.push_back(0.0);
  }

  RR = BB;
  RR2 = BB;

  double initres = sqrt(dot(RR, RR));
  double eps = max(mineps, tol * initres);
  double res = initres;
  for (I = 0; I < maxit; I++) {
    if (info)
      if (rank == 0) {
        cout << "It: " << I << " res = " << scientific << res
             << " tol = " << scientific << res / initres << endl;
      }
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

    Update(PP);
    SpMV(DD, PP, PP2);
    Update(PP2);
    SpMV(_mtr, PP2, VV);
    alphai = dot(RR2, VV);
    if (fabs(alphai) < RhoMin) return -3;
    alphai = Rhoi_1 / alphai;
    SS = RR;
    axpby(SS, VV, 1.0, -alphai);
    Update(SS);
    SpMV(DD, SS, SS2);
    Update(SS2);
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
  if (info && rank == 0) {
    cout << "Solver_BiCGSTAB: outres:" << scientific << res << endl;
    cout << "Tol: " << scientific << res / initres << endl;
  }

  return I;
}

int main(int argc, char **argv) {
  // for tests
  int flag = 1;
  double max1 = 0, max2 = 0, max3 = 0, max_solv = 0;

  int Nx, Ny, Nz, N;
  int numproces, proc_rank;

  Nx = atoi(argv[1]);
  Ny = atoi(argv[2]);
  Nz = atoi(argv[3]);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numproces);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int *Glob2Loc;
  Glob2Loc = new int[Nx * Ny * Nz];

  int periods[] = {0, 0, 0};
  int dims[] = {atoi(argv[4]), atoi(argv[4]), atoi(argv[4])};
  int coords[3];

  // initialization
  for (int i = 0; i < Nx * Ny * Nz; i++) Glob2Loc[i] = -1;

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm);
  controlMatrix = GenerateMatrix1(Nx, Ny, Nz, atoi(argv[4]));

  // fill in the Glob2Loc local numbers of their cells
  for (int i = 0; i < Rows.size(); i++) {
    Glob2Loc[Rows[i]] = i;
  }
  rsv.resize(numproces);
  snd.resize(numproces);
  int _temp;

  // the formation of messages to the receive(halo) and to send
  for (int k = 0; k < controlMatrix.indexRow.size() - 1; k++) {
    for (int i = controlMatrix.indexRow[k]; i < controlMatrix.indexRow[k + 1];
         i++) {
      if (Glob2Loc[controlMatrix.col[i]] == -1) {
        _temp = Part[controlMatrix.col[i]];
        rsv[_temp].push_back(controlMatrix.col[i]);
        snd[_temp].push_back(Rows[k]);
      }
    }
  }
  for (int i = 0; i < snd.size(); i++) {
    if (snd[i].size() > 0) {
      nbProc.push_back(i);
    }
  }
  sort(nbProc.begin(), nbProc.end());
  for (int i = 0; i < rsv.size(); i++) {
    if (rsv[i].size() > 0) {
      for (int j = 0; j < rsv[i].size(); j++) {
        Halo.push_back(rsv[i][j]);
        bufRecv.push_back(rsv[i][j]);
      }
    }
  }
  for (int i = 0; i < snd.size(); i++) {
    if (snd[i].size() > 0) {
      for (int j = 0; j < snd[i].size(); j++) {
        bufSend.push_back(snd[i][j]);
      }
    }
  }

  for (int i = 0; i < bufSend.size(); i++) {
    for (int j = 0; j < Rows.size(); j++) {
      if (bufSend[i] == Rows[j]) bufSend[i] = j;
    }
  }
  for (int i = 0; i < snd.size(); i++) {
    if (snd[i].size() > 0) {
      for (int j = 0; j < snd[i].size(); ++j) {
        for (int r = 0; r < Rows.size(); r++) {
          if (snd[i][j] == Rows[r]) snd[i][j] = r;
        }
      }
    }
  }
  // entry in Glob2Loc local numbers for the halo cells
  for (int i = 0; i < Halo.size(); i++)
    // Glob2Loc[Halo[i]] = i + (controlMatrix.col.size() - bufRecv.size());
    Glob2Loc[Halo[i]] = i + Rows.size();

  // translation column in JA in local numbering
  for (int k = 0; k < controlMatrix.indexRow.size() - 1; k++) {
    for (int i = controlMatrix.indexRow[k]; i < controlMatrix.indexRow[k + 1];
         i++) {
      controlMatrix.col[i] = Glob2Loc[controlMatrix.col[i]];
    }
  }
  // translation buffer messages in local numbering
  for (int i = 0; i < bufRecv.size(); i++) {
    bufRecv[i] = Glob2Loc[bufRecv[i]];
  }
  double t_b = MPI_Wtime();
  int r = BicGSTAB(controlMatrix, atof(argv[5]), atoi(argv[6]));
  double t_e = MPI_Wtime() - t_b;
  if (rank != 0) {
    MPI_Send(&t_e, 1, MPI_DOUBLE, 0, rank, comm);
  }
  if (rank == 0) {
    if (t_e > max_solv) max_solv = t_e;
    for (int i = 0; i < numproces - 1; ++i) {
      MPI_Recv(&t_e, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
      if (t_e > max_solv) max_solv = t_e;
    }
  }
  ofstream fout("result", ios::app);
  if (rank == 0) {
    fout << "BicGSTAB :" << t_e << endl;
  }
  if (flag == 1) {
    MPI_Status status;
    double max1 = 0.0, max2 = 0.0, max3 = 0.0;
    vector<double> X, Y;
    int some_val = controlMatrix.N + Halo.size();
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
    double time = MPI_Wtime();
    dot_result = dot(X, Y);
    time = MPI_Wtime() - time;

    if (rank != 0) {
      MPI_Send(&time, 1, MPI_DOUBLE, 0, rank, comm);
    }
    if (rank == 0) {
      if (time > max1) max1 = time;
      for (int i = 0; i < numproces - 1; ++i) {
        MPI_Recv(&time, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm,
                 &status);
        if (time > max1) max1 = time;
      }
    }
    double time1 = MPI_Wtime();
    axpby(X, Y, 3.0, 3.0);
    time1 = MPI_Wtime() - time1;
    if (rank != 0) {
      MPI_Send(&time1, 1, MPI_DOUBLE, 0, rank, comm);
    }
    if (rank == 0) {
      if (time1 > max2) max2 = time1;
      cout << "max0" << max2 << endl;
      for (int i = 0; i < numproces - 1; ++i) {
        MPI_Recv(&time1, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm,
                 &status);
        if (time1 > max2) max2 = time1;
      }
    }
    double time2 = MPI_Wtime();
    SpMV(controlMatrix, Y, Y);
    time2 = MPI_Wtime() - time2;

    if (rank != 0) {
      MPI_Send(&time2, 1, MPI_DOUBLE, 0, rank, comm);
    }
    if (rank == 0) {
      if (time2 > max3) max3 = time2;
      for (int i = 0; i < numproces - 1; ++i) {
        MPI_Recv(&time2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm,
                 &status);
        if (time2 > max3) max3 = time2;
      }
    }
    if (rank == 0) {
      fout << "dot :" << max1 << endl;
      fout << "axpby :" << max2 << endl;
      fout << "SPMV :" << max3 << endl;
    }
  }
  fout.close();
  MPI_Finalize();
  return 0;
}
