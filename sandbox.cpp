/*
 *  This file is a part of Libint.
 *  Copyright (C) 2004-2020 Edward F. Valeev
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 */

// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Libint Gaussian integrals library
#include <libint2.hpp>

// BTAS include
#include <btas/btas.h>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                 // to meet the layout of the integrals returned by the Libint integral library

std::vector<libint2::Atom> read_geometry(const std::string& filename);
std::vector<libint2::Shell> make_sto3g_basis(const std::vector<libint2::Atom>& atoms);
std::vector<libint2::Shell> make_cc_pvdz_basis(const std::vector<libint2::Atom>& atoms);

Matrix compute_soad(const std::vector<libint2::Atom>& atoms);

Matrix compute_1body_ints(const libint2::BasisSet& obs,
                          libint2::Operator obtype,
                          const std::vector<libint2::Atom>& atoms = std::vector<libint2::Atom>());

Matrix compute_2body_fock_simple(const libint2::BasisSet& obs,
                                 const Matrix& D);

Matrix compute_2body_fock(const libint2::BasisSet& obs,
                                 const Matrix& D);

double mp2_energy(const btas::Tensor<double>& ia_jb, int nocc, int n, const Eigen::VectorXd& evals);
btas::Tensor<double> make_v_gg_gg(const Matrix& C, const btas::Tensor<double>& T);
btas::Tensor<double> make_v_aa_aa(const libint2::BasisSet& obs);

btas::Tensor<double>
calcResidual(const btas::Tensor<double>& v_oo_uu, const btas::Tensor<double>& v_uu_uu, const btas::Tensor<double>& t,
             const btas::Tensor<double>& I_u_u, const btas::Tensor<double>& I_o_o, const btas::Tensor<double>& I_oo_oo,
             const btas::Tensor<double>& I_ou_ou, const btas::Tensor<double>& I_uo_ou, const int n, const int nocc);
btas::Tensor<double> make_I_uo_ou(const btas::Tensor<double> &v_uo_ou, const btas::Tensor<double> &v_uu_oo,
                                  const btas::Tensor<double> &v_ou_ou, const btas::Tensor<double> &t_oo_uu,
                                  const int nocc, const int n);
btas::Tensor<double> make_I_ou_ou(const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n);
btas::Tensor<double> make_I_oo_oo(const btas::Tensor<double>& v_oo_oo, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n);
btas::Tensor<double> make_I_o_o(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n);
btas::Tensor<double> make_I_u_u(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n);
double calc_ccd_energy(btas::Tensor<double> const & t_oo_uu, btas::Tensor<double> const & v_oo_uu, int n, int nocc);
btas::Tensor<double> ccd_permute(btas::Tensor<double>& tensor);
//btas::Tensor<double> /*void*/ ccd_permute(btas::Tensor<double>& tensor);
btas::Tensor<double> make_F_mo(const Matrix& C, const btas::Tensor<double>& F);

/// CCSD Intermediates
btas::Tensor<double> make_h_j_i(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau);
btas::Tensor<double> make_h_b_i(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& singles_amps);
btas::Tensor<double> make_h_b_a(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau);
btas::Tensor<double> make_g_j_i(const btas::Tensor<double>& h_j_i, const btas::Tensor<double>& fock_matrix,
                                const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_ou_oo);
btas::Tensor<double> make_g_b_a(const btas::Tensor<double>& h_b_a, const btas::Tensor<double>& fock_matrix,
                                const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_uo);
btas::Tensor<double> make_a_uv_ij(const btas::Tensor<double>& v_oo_oo, const btas::Tensor<double>& v_ou_oo,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uo_oo,
                                  const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau);
btas::Tensor<double> make_b_ab_BY(const btas::Tensor<double>& v_uu_uu, const btas::Tensor<double>& v_uu_uo,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_ou);
btas::Tensor<double> make_j_ua_Bi(const btas::Tensor<double>& v_ou_uo, const btas::Tensor<double>& v_ou_oo,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_uo,
                                  const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& T, const btas::Tensor<double>& doubles_amps);
btas::Tensor<double> make_k_ua_iB(const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_ou,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_oo,
                                  const btas::Tensor<double>& T);
btas::Tensor<double> t1_residual(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& singles_amps,
                                 const btas::Tensor<double>& h_b_a, const btas::Tensor<double>& h_j_i, const btas::Tensor<double>& h_b_i,
                                 const btas::Tensor<double>& doubles_amps, const btas::Tensor<double>& v_uo_ou,
                                 const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_ou, const btas::Tensor<double>& tau,
                                 const btas::Tensor<double>& v_uo_oo, const btas::Tensor<double>& v_uu_oo, int nuocc, int nocc);
btas::Tensor<double> t2_residual(const btas::Tensor<double>& v_oo_uu, const btas::Tensor<double>& a_oo_oo,
                                 const btas::Tensor<double>& tau, const btas::Tensor<double>& b_uu_uu, const btas::Tensor<double>& g_u_u,
                                 const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& g_o_o,
                                 const btas::Tensor<double>& v_ou_uu, const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_oo_uo,
                                 const btas::Tensor<double>& v_ou_uo, const btas::Tensor<double>& j_ou_uo, const btas::Tensor<double>& k_ou_ou,
                                 const btas::Tensor<double>& doubles_amps, int nuocc, int nocc);
btas::Tensor<double> make_tau(const btas::Tensor<double>& doubles_amps, const btas::Tensor<double>& singles_amps);
btas::Tensor<double> make_T(const btas::Tensor<double>& doubles_amps, const btas::Tensor<double>& singles_amps);
double calc_ccsd_energy(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& singles_amplitudes,
                        const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau);

int main(int argc, char *argv[]) {

  using std::cout;
  using std::cerr;
  using std::endl;

  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Atom;

  try {

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    // read geometry from a file; by default read from h2o.xyz, else take filename (.xyz) from the command line
    const auto filename = (argc > 1) ? argv[1] : "h2.xyz"; // h2o.xyz
    std::vector<libint2::Atom> atoms = read_geometry(filename);
	const auto basisname = (argc > 2) ? argv[2] : "sto-3g"; //sto-3g //aug-cc-pVDZ

    // count the number of electrons
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i)
      nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / 2;

    // compute the nuclear repulsion energy
    auto enuc = 0.0;
    for (auto i = 0; i < atoms.size(); i++)
      for (auto j = i + 1; j < atoms.size(); j++) {
        auto xij = atoms[i].x - atoms[j].x;
        auto yij = atoms[i].y - atoms[j].y;
        auto zij = atoms[i].z - atoms[j].z;
        auto r2 = xij*xij + yij*yij + zij*zij;
        auto r = sqrt(r2);
        enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
      }
    cout << "\tNuclear repulsion energy = " << enuc << endl;

    /*** =========================== ***/
    /*** create basis set            ***/
    /*** =========================== ***/

    BasisSet obs(basisname, atoms);

    size_t nao = 0;
    for (auto s=0; s<obs.size(); ++s)
      nao += obs[s].size();

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize();

    // compute overlap integrals
    auto S = compute_1body_ints(obs, Operator::overlap, atoms);

    cout << "\n\tOverlap Integrals:\n";
    cout << S << endl;

    // compute kinetic-energy integrals
    auto T = compute_1body_ints(obs, Operator::kinetic, atoms);
    cout << "\n\tKinetic-Energy Integrals:\n";
    cout << T << endl;

    // compute nuclear-attraction integrals
    Matrix V = compute_1body_ints(obs, Operator::nuclear, atoms);
    cout << "\n\tNuclear Attraction Integrals:\n";
    cout << V << endl;

    // Core Hamiltonian = T + V
    Matrix H = T + V;
    cout << "\n\tCore Hamiltonian:\n";
    cout << H << endl;

    // T and V no longer needed, free up the memory
    T.resize(0,0);
    V.resize(0,0);

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    const auto use_hcore_guess = basisname == "sto-3g" ? false : true;

    Matrix D;
    Matrix C;
    Eigen::VectorXd eps;
    if (use_hcore_guess) { // hcore guess
      // solve H C = e S C
      Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
      auto eps = gen_eig_solver.eigenvalues();
      C = gen_eig_solver.eigenvectors();
      cout << "\n\tInitial C Matrix:\n";
      cout << C << endl;

      // compute density, D = C(occ) . C(occ)T
      auto C_occ = C.leftCols(ndocc);
      D = C_occ * C_occ.transpose();
    }
    else {  // SOAD as the guess density, assumes STO-nG basis
      D = compute_soad(atoms);
    }

    cout << "\n\tInitial Density Matrix:\n";
    cout << D << endl;

    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/

    const auto maxiter = 300;
    const auto conv = 1e-10;
    auto iter = 0;
    auto rmsd = 0.0;
    auto ediff = 0.0;
    auto ehf = 0.0;
    //auto emp2 = 0.0; // MP2 energy
    Matrix F;
    do {
      const auto tstart = std::chrono::high_resolution_clock::now();
      ++iter;

      // Save a copy of the energy and the density
      auto ehf_last = ehf;
      auto D_last = D;

      // build a new Fock matrix
      F = H;//auto
      F += compute_2body_fock(obs, D);

      if (iter == 1) {
        cout << "\n\tFock Matrix:\n";
        cout << F << endl;
      }

      // solve F C = e S C
      Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
      C = gen_eig_solver.eigenvectors();
      eps = gen_eig_solver.eigenvalues();

      // compute density, D = C(occ) . C(occ)T
      auto C_occ = C.leftCols(ndocc);
      D = C_occ * C_occ.transpose();

      // compute HF energy
      ehf = 0.0;
      for (auto i = 0; i < nao; i++)
        for (auto j = 0; j < nao; j++)
            ehf += D(i, j) * (H(i, j) + F(i, j));

      // compute difference with last iteration
      ediff = ehf - ehf_last;
      rmsd = (D - D_last).norm();

      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;

      if (iter == 1)
        std::cout <<
        "\n\n Iter        E(elec)              E(tot)               Delta(E)             RMS(D)         Time(s)\n";
      printf(" %02d %20.12f %20.12f %20.12f %20.12f %10.5lf\n", iter, ehf, ehf + enuc,
             ediff, rmsd, time_elapsed.count());

    } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));

    /*** ============ ***/
    /***   CC prep    ***/
    /*** ============ ***/

    btas::Tensor<double> v_aa_aa = make_v_aa_aa(obs);
    auto v_gg_gg = make_v_gg_gg(C, v_aa_aa);

    auto nuocc = nao - ndocc;
    auto n = nao;
    std::cout << "N: " << n << " Ndocc: " << ndocc << " nuocc: " << nuocc << std::endl;

    //std::cout << "v_oo_oo:\n";
    btas::Tensor<double> v_oo_oo(ndocc, ndocc, ndocc, ndocc);
    for (int i = 0; i != ndocc; ++i) {
        for (int j = 0; j != ndocc; ++j) {
            for (int k = 0; k != ndocc; ++k) {
                for (int l = 0; l != ndocc; ++l) {
                    v_oo_oo(i, j, k, l) = v_gg_gg(i, j, k, l);
                    //std::cout << v_oo_oo(i, j, k, l) << " \n";
                }
            }
        }
    }

    //std::cout << "v_uu_uu:\n";
    btas::Tensor<double> v_uu_uu(nuocc, nuocc, nuocc, nuocc);
    for (int a = 0; a != nuocc; ++a) {
        for (int b = 0; b != nuocc; ++b) {
            for (int c = 0; c != nuocc; ++c) {
                for (int d = 0; d != nuocc; ++d) {
                    v_uu_uu(a, b, c, d) = v_gg_gg(a + ndocc, b + ndocc, c + ndocc, d + ndocc);
                    //std::cout << v_uu_uu(a, b, c, d) << " \n";
                }
            }
        }
    }

    //std::cout << "v_uu_oo:\n";
    btas::Tensor<double> v_uu_oo(nuocc, nuocc, ndocc, ndocc);
    for (int a = 0; a != nuocc; ++a) {
        for (int b = 0; b != nuocc; ++b) {
            for (int i = 0; i != ndocc; ++i) {
                for (int j = 0; j != ndocc; ++j) {
                    v_uu_oo(a, b, i, j) = v_gg_gg(a + ndocc, b + ndocc, i, j);
                    //std::cout << v_uu_oo(a, b, i, j) << " \n";
                }
            }
        }
    }

    //std::cout << "v_oo_uu:\n";
    btas::Tensor<double> v_oo_uu(ndocc, ndocc, nuocc, nuocc);
    for (int i = 0; i != ndocc; ++i) {
        for (int j = 0; j != ndocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                   v_oo_uu(i, j, a, b) = v_gg_gg(i, j, a + ndocc, b + ndocc);
                   //std::cout << v_oo_uu(i, j, a, b) << " \n";
                }
            }
        }
    }

    //std::cout << "v_ou_ou:\n";
    btas::Tensor<double> v_ou_ou(ndocc, nuocc, ndocc, nuocc);
    for (int i = 0; i != ndocc; ++i) {
        for (int a = 0; a != nuocc; ++a) {
            for (int j = 0; j != ndocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    v_ou_ou(i, a, j, b) = v_gg_gg(i, a + ndocc, j, b + ndocc);
                    //std::cout << v_ou_ou(i, a, j, b) << " \n";
                }
            }
        }
    }

    //std::cout << "v_uo_ou:\n";
    btas::Tensor<double> v_uo_ou(nuocc, ndocc, ndocc, nuocc);
    for (int a = 0; a != nuocc; ++a) {
        for (int i = 0; i != ndocc; ++i) {
            for (int j = 0; j != ndocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    v_uo_ou(a, i, j, b) = v_gg_gg(a + ndocc, i, j, b + ndocc);
                    //std::cout << v_uo_ou(a, i, j, b) << " \n";
                }
            }
        }
    }

    // v_ou_oo
    btas::Tensor<double> v_ou_oo(ndocc, nuocc, ndocc, ndocc);
    for (size_t i = 0; i != ndocc; ++i) {
        for (size_t a = 0; a != nuocc; ++a) {
            for (size_t j = 0; j != ndocc; ++j) {
                for (size_t k = 0; k != ndocc; ++k) {
                    v_ou_oo(i, a, j, k) = v_gg_gg(i, a + ndocc, j, k);
                }
            }
        }
    }

    // v_uu_uo
    btas::Tensor<double> v_uu_uo(nuocc, nuocc, nuocc, ndocc);
    for (size_t a = 0; a != nuocc; ++a) {
        for (size_t b = 0; b != nuocc; ++b) {
            for (size_t c = 0; c != nuocc; ++c) {
                for (size_t i = 0; i != ndocc; ++i) {
                    v_uu_uo(a, b, c, i) = v_gg_gg(a + ndocc, b + ndocc, c + ndocc, i);
                }
            }
        }
    }

    //v_uo_oo
    btas::Tensor<double> v_uo_oo(nuocc, ndocc, ndocc, ndocc);
    for (size_t a = 0; a != nuocc; ++a) {
        for (size_t i = 0; i != ndocc; ++i) {
            for (size_t j = 0; j != ndocc; ++j) {
                for (size_t k = 0;  k != ndocc; ++k) {
                    v_uo_oo(a, i, j, k) = v_gg_gg(a + ndocc, i, j, k);
                }
            }
        }
    }

    // v_uu_ou
    btas::Tensor<double> v_uu_ou(nuocc, nuocc, ndocc, nuocc);
    for (size_t a = 0; a != nuocc; ++a) {
        for (size_t b = 0; b != nuocc; ++b) {
            for (size_t i = 0; i != ndocc; ++i) {
                for (size_t c = 0; c != nuocc; ++c) {
                    v_uu_ou(a, b, i, c) = v_gg_gg(a + ndocc, b + ndocc, i, c + ndocc);
                }
            }
        }
    }

    // v_ou_uo
    btas::Tensor<double> v_ou_uo(ndocc, nuocc, nuocc, ndocc);
    for (size_t i = 0; i != ndocc; ++i) {
        for (size_t a = 0; a != nuocc; ++a) {
            for (size_t j = 0; j != ndocc; ++j) {
                for (size_t b = 0; b != nuocc; ++b) {
                    v_ou_uo(i, a, j, b) = v_gg_gg(i, a + ndocc, j, b + ndocc);
                }
            }
        }
    }

    // v_ou_uu
    btas::Tensor<double> v_ou_uu(ndocc, nuocc, nuocc, nuocc);
    for (size_t i = 0; i != ndocc; ++i) {
        for (size_t a = 0; a != nuocc; ++a) {
            for (size_t b = 0; b != ndocc; ++b) {
                for (size_t c = 0; c != nuocc; ++c) {
                    v_ou_uu(i, a, b, c) = v_gg_gg(i, a + ndocc, b + ndocc, c + ndocc);
                }
            }
        }
    }

    // v_oo_uo
    btas::Tensor<double> v_oo_uo(ndocc, ndocc, nuocc, ndocc);
    for (size_t i = 0; i != ndocc; ++i) {
        for (size_t j = 0; j != ndocc; ++j) {
            for (size_t a = 0; a != nuocc; ++a) {
                for (size_t k = 0; k != ndocc; ++k) {
                    v_ou_uo(i, j, a, k) = v_gg_gg(i, j, a + ndocc, k);
                }
            }
        }
    }

    const int max_cc_iter = 300;
    int cc_iter = 0;
    double ccd_energy = 0.0;
    double ccd_energy_last = 0.0;
    double e_change;
    btas::Tensor<double> t_oo_uu_prev;
    btas::Tensor<double> t_o_u_prev;
    rmsd = 0.0; //TODO: make independent variable for this?
    btas::Tensor<double> t_oo_uu(ndocc, ndocc, nuocc, nuocc);
    btas::Tensor<double> t_o_u(ndocc, nuocc);
    t_oo_uu.fill(0);
    t_o_u.fill(0);
    btas::Tensor<double> I_uo_ou(nuocc, ndocc, ndocc, nuocc);
    btas::Tensor<double> I_ou_ou(ndocc, nuocc, ndocc, nuocc);
    btas::Tensor<double> I_oo_oo(ndocc, ndocc, ndocc, ndocc);
    btas::Tensor<double> I_o_o(ndocc, ndocc);
    btas::Tensor<double> I_u_u(nuocc, nuocc);

    double ccsd_energy = 0.0;
    double ccsd_energy_last = 0.0;
    btas::Tensor<double> h_j_i(ndocc, ndocc);
    btas::Tensor<double> h_b_i(nuocc, ndocc);
    btas::Tensor<double> h_b_a(nuocc, nuocc);
    btas::Tensor<double> g_j_i(ndocc, ndocc);
    btas::Tensor<double> g_b_a(nuocc, nuocc);
    btas::Tensor<double> a_uv_ij(ndocc, ndocc, ndocc, ndocc);
    btas::Tensor<double> b_ab_BY(nuocc, nuocc, nuocc, nuocc);
    btas::Tensor<double> j_ua_Bi(ndocc, nuocc, ndocc, nuocc);
    btas::Tensor<double> k_ua_iB(ndocc, nuocc, ndocc, nuocc);
    btas::Tensor<double> tau;
    btas::Tensor<double> Tint;
    btas::Tensor<double> R1;
    btas::Tensor<double> R2;

    btas::Tensor<double> Ften(n, n);
    for (int i = 0; i != n; i++) {
        for (int j = 0; j != n; j++) {
            Ften(i, j) = F(i, j);
        }
    }
    //std::cout << "F_mo:\n";
    auto F_mo = make_F_mo(C, Ften);
    for (int i = 0; i != n; i++) {
        for (int j = 0; j != n; j++) {
            //std::cout << F_mo(i,j) << "\t";
        }
        //std::cout << std::endl;
    }

    /*** ============== ***/
    /***    CCD loop    ***/
    /*** ============== ***/
    std::cout << "Up to ccsd loop\n";
    do {
        ++cc_iter;
        t_oo_uu_prev = t_oo_uu;
        ccd_energy_last = ccd_energy;

        I_uo_ou = make_I_uo_ou(v_uo_ou, v_uu_oo, v_ou_ou, t_oo_uu, ndocc, n);
        I_ou_ou = make_I_ou_ou(v_ou_ou, v_uu_oo, t_oo_uu, ndocc, n);
        I_oo_oo = make_I_oo_oo(v_oo_oo, v_uu_oo, t_oo_uu, ndocc, n);
        I_o_o = make_I_o_o(v_uu_oo, t_oo_uu, ndocc, n);
        I_u_u = make_I_u_u(v_uu_oo, t_oo_uu, ndocc, n);
        //calcResidual(v_oo_uu, v_uu_uu, t_oo_uu, I_u_u, I_o_o, I_oo_oo, I_ou_ou, I_uo_ou, n, ndocc); // = R residual

        tau = make_tau(t_oo_uu, t_o_u);
        Tint = make_T(t_oo_uu, t_o_u);
        h_b_a = make_h_b_a(F_mo, v_uu_oo, tau);
        h_b_i = make_h_b_i(F_mo, v_uu_oo, t_o_u);
        h_j_i = make_h_j_i(F_mo, v_uu_oo, tau);
        g_j_i = make_g_j_i(h_j_i, F_mo, t_o_u, v_ou_oo);
        g_b_a = make_g_b_a(h_b_a, F_mo, t_o_u, v_uu_uo);
        a_uv_ij = make_a_uv_ij(v_oo_oo, v_ou_oo, t_o_u, v_uo_oo, v_uu_oo, tau);
        b_ab_BY = make_b_ab_BY(v_uu_uu, v_uu_uo, t_o_u, v_uu_ou);
        j_ua_Bi = make_j_ua_Bi(v_ou_uo, v_ou_oo, t_o_u, v_uu_uo, v_uu_oo, Tint, t_oo_uu);
        k_ua_iB = make_k_ua_iB(v_ou_ou, v_uu_ou, t_o_u, v_uu_oo, Tint);

        R1 = t1_residual(F_mo, t_o_u, h_b_a, h_j_i, h_b_i, t_oo_uu, v_uo_ou, v_ou_ou, v_uu_ou, tau, v_uo_oo, v_uu_oo, nuocc, ndocc);
        R2 = t2_residual(v_oo_uu, a_uv_ij, tau, b_ab_BY, g_b_a, t_o_u, g_j_i, v_ou_uu, v_ou_ou, v_oo_uo, v_ou_uo, j_ua_Bi, k_ua_iB, t_oo_uu, nuocc, ndocc);

        // update amplitudes
        for (int i = 0; i != ndocc; ++i) {
            for (int j = 0; j != ndocc; ++j) {
                for (int a = 0; a != nuocc; ++a) {
                    for (int b = 0; b != nuocc; ++b) { /// TODO
                        t_oo_uu(i, j, a, b) +=
                                R2(i, j, a, b) / (F_mo(i, i) + F_mo(j, j) - F_mo(a + ndocc, a + ndocc) - F_mo(b + ndocc, b + ndocc));
                    }
                }
            }
        }
        for (int i = 0; i != ndocc; ++i) {
            for (int a = 0; a != ndocc; ++a) {
                t_o_u(i, a) +=
                        R1(i, a) / (F_mo(i, i) - F_mo(a + ndocc, a + ndocc));
            }
        }

        /*if (cc_iter == 1) {
            for (int i = 0; i != ndocc; ++i) {
                for (int j = 0; j != ndocc; ++j) {
                    for (int a = 0; a != nuocc; ++a) {
                        for (int b = 0; b != nuocc; ++b) {
                            if (v_oo_uu(i, j, a, b) != R(i, j, a, b)) {
                                //std::cout << "*ijab*:" << i << j << a << b << " v=" << v_oo_uu(i, j, a, b) << " R="
                                          //<< R(i, j, a, b) << std::endl;
                            }
                        }
                    }
                }
            }
        }*/

/*        ccd_energy = calc_ccd_energy(t_oo_uu, v_oo_uu, n, ndocc);
        std::cout << "CCD energy for iteration " << cc_iter << ": " << ccd_energy << std::endl;*/

        ccsd_energy = calc_ccsd_energy(F_mo, t_o_u, v_uu_oo, tau);
        std::cout << "CCSD energy for iteration " << cc_iter << ": " << ccsd_energy << std::endl;

/*        e_change = ccd_energy - ccd_energy_last;
        std::cout << "Energy change for iteration " << cc_iter << ": " << e_change << std::endl;*/
        e_change = ccsd_energy - ccsd_energy_last;
        std::cout << "Energy change for iteration " << cc_iter << ": " << e_change << std::endl;
    } while (std::abs(e_change) > conv && cc_iter != max_cc_iter);

    printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);
    //printf("** MP2 energy = %20.12f\n", emp2);
    //printf("** Total MP2 energy = %20.12f\n", ehf + enuc + emp2);

    libint2::finalize(); // done with libint

  } // end of try block; if any exceptions occurred, report them and exit cleanly

  catch (const char* ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  }
  catch (std::string& ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  }
  catch (std::exception& ex) {
    cerr << ex.what() << endl;
    return 1;
  }
  catch (...) {
    cerr << "caught unknown exception\n";
    return 1;
  }

  return 0;
}

// this reads the geometry in the standard xyz format supported by most chemistry software
std::vector<libint2::Atom> read_dotxyz(std::istream& is) {
  // line 1 = # of atoms
  size_t natom;
  is >> natom;
  // read off the rest of line 1 and discard
  std::string rest_of_line;
  std::getline(is, rest_of_line);

  // line 2 = comment (possibly empty)
  std::string comment;
  std::getline(is, comment);

  std::vector<libint2::Atom> atoms(natom);
  for (auto i = 0; i < natom; i++) {
    std::string element_label;
    double x, y, z;
    is >> element_label >> x >> y >> z;

    // .xyz files report element labels, hence convert to atomic numbers
    int Z;
    if (element_label == "H")
      Z = 1;
    else if (element_label == "C")
      Z = 6;
    else if (element_label == "N")
      Z = 7;
    else if (element_label == "O")
      Z = 8;
    else if (element_label == "F")
      Z = 9;
    else if (element_label == "S")
      Z = 16;
    else if (element_label == "Cl")
      Z = 17;
    else {
      std::cerr << "read_dotxyz: element label \"" << element_label << "\" is not recognized" << std::endl;
      throw "Did not recognize element label in .xyz file";
    }

    atoms[i].atomic_number = Z;

    // .xyz files report Cartesian coordinates in angstroms; convert to bohr
    const auto angstrom_to_bohr = 1 / 0.52917721092; // 2010 CODATA value
    atoms[i].x = x * angstrom_to_bohr;
    atoms[i].y = y * angstrom_to_bohr;
    atoms[i].z = z * angstrom_to_bohr;
  }

  return atoms;
}

std::vector<libint2::Atom> read_geometry(const std::string& filename) {

  std::cout << "Will read geometry from " << filename << std::endl;
  std::ifstream is(filename);
  assert(is.good());

  // to prepare for MPI parallelization, we will read the entire file into a string that can be
  // broadcast to everyone, then converted to an std::istringstream object that can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise throw an exception
  if ( filename.rfind(".xyz") != std::string::npos)
    return read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}

size_t nbasis(const libint2::BasisSet& basissets) {
  size_t n = 0;
  for (const auto& shell: basissets)
    n += shell.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::BasisSet>& basissets) {
  size_t n = 0;
  for (auto basisset: basissets)
    n = std::max(basisset.max_nprim(), n);
  return n;
}

int max_l(const std::vector<libint2::BasisSet>& basissets) {
  int l = 0;
  for (auto basisset: basissets)
      l = std::max(l, int(basisset.max_l()));
  return l;
}

// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<libint2::Atom>& atoms) {

  // compute number of atomic orbitals
  size_t nao = 0;
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) // H, He
      nao += 1;
    else if (Z <= 10) // Li - Ne
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported";
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) { // H, He
      D(ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    }
    else if (Z <= 10) {
      D(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      D(ao_offset+1, ao_offset+1) = (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double)(Z - 4)/3 : 0;
      for(auto xyz=0; xyz!=3; ++xyz)
        D(ao_offset+2+xyz, ao_offset+2+xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return D * 0.5; // we use densities normalized to # of electrons/2
}

Matrix compute_1body_ints(const libint2::BasisSet& obs,
                          libint2::Operator obtype,
                          const std::vector<libint2::Atom>& atoms)
{
  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = nbasis(obs);
  Matrix result(n,n);

  // construct the overlap integrals engine
  Engine engine(obtype, max_nprim(obs), max_l(obs), 0);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == Operator::nuclear) {
    std::vector<std::pair<double,std::array<double,3>>> q;
    for(const auto& atom : atoms) {
      q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
    }
    engine.set_params(q);
  }

  auto shell2bf = obs.shell2bf();

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over unique shell pairs, {s1,s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
  for(auto s1=0; s1!=obs.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = obs[s2].size();

      // compute shell pair; return is the pointer to the buffer
      engine.compute(obs[s1], obs[s2]);

      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
      if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
      result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

    }
  }

  return result;
}

Matrix compute_2body_fock_simple(const libint2::BasisSet& obs,
                                 const Matrix& D) {

  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = nbasis(obs);
  Matrix G = Matrix::Zero(n,n);

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(obs), max_nprim(obs), 0);

  auto shell2bf = obs.shell2bf();

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over shell pairs of the Fock matrix, {s1,s2}
  // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
  for(auto s1=0; s1!=obs.size(); ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();

    for(auto s2=0; s2!=obs.size(); ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      // loop over shell pairs of the density matrix, {s3,s4}
      // again symmetry is not used for simplicity
      for(auto s3=0; s3!=obs.size(); ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        for(auto s4=0; s4!=obs.size(); ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          // we don't have an analog of Eigen for tensors (yet ... see github.com/BTAS/BTAS, under development)
          // hence some manual labor here:
          // 1) loop over every integral in the shell set (= nested loops over basis functions in each shell)
          // and 2) add contribution from each integral
          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;
                  G(bf1,bf2) += D(bf3,bf4) * 2.0 * buf_1234[f1234];
                }
              }
            }
          }

          // exchange contribution to the Fock matrix is from {s1,s3,s2,s4} integrals
          engine.compute(obs[s1], obs[s3], obs[s2], obs[s4]);
          const auto* buf_1324 = buf[0];

          for(auto f1=0, f1324=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f3=0; f3!=n3; ++f3) {
              const auto bf3 = f3 + bf3_first;
              for(auto f2=0; f2!=n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1324) {
                  const auto bf4 = f4 + bf4_first;
                  G(bf1,bf2) -= D(bf3,bf4) * buf_1324[f1324];
                }
              }
            }
          }

        }
      }
    }
  }

  return G;
}

Matrix compute_2body_fock(const libint2::BasisSet& obs,
                          const Matrix& D) {

  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;

  std::chrono::duration<double> time_elapsed = std::chrono::duration<double>::zero();

  const auto n = nbasis(obs);
  Matrix G = Matrix::Zero(n,n);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(obs), max_l(obs), 0);

  auto shell2bf = obs.shell2bf();

  const auto& buf = engine.results();

  // The problem with the simple Fock builder is that permutational symmetries of the Fock,
  // density, and two-electron integrals are not taken into account to reduce the cost.
  // To make the simple Fock builder efficient we must rearrange our computation.
  // The most expensive step in Fock matrix construction is the evaluation of 2-e integrals;
  // hence we must minimize the number of computed integrals by taking advantage of their permutational
  // symmetry. Due to the multiplicative and Hermitian nature of the Coulomb kernel (and realness
  // of the Gaussians) the permutational symmetry of the 2-e ints is given by the following relations:
  //
  // (12|34) = (21|34) = (12|43) = (21|43) = (34|12) = (43|12) = (34|21) = (43|21)
  //
  // (here we use chemists' notation for the integrals, i.e in (ab|cd) a and b correspond to
  // electron 1, and c and d -- to electron 2).
  //
  // It is easy to verify that the following set of nested loops produces a permutationally-unique
  // set of integrals:
  // foreach a = 0 .. n-1
  //   foreach b = 0 .. a
  //     foreach c = 0 .. a
  //       foreach d = 0 .. (a == c ? b : c)
  //         compute (ab|cd)
  //
  // The only complication is that we must compute integrals over shells. But it's not that complicated ...
  //
  // The real trick is figuring out to which matrix elements of the Fock matrix each permutationally-unique
  // (ab|cd) contributes. STOP READING and try to figure it out yourself. (to check your answer see below)

  // loop over permutationally-unique set of shells
  for(auto s1=0; s1!=obs.size(); ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = obs[s1].size();   // number of basis functions in this shell

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      for(auto s3=0; s3<=s1; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for(auto s4=0; s4<=s4_max; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
          auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

          const auto tstart = std::chrono::high_resolution_clock::now();

          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          const auto tstop = std::chrono::high_resolution_clock::now();
          time_elapsed += tstop - tstart;

          // ANSWER
          // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
          //    F(a,b) += (ab|cd) * D(c,d)
          //    F(c,d) += (ab|cd) * D(a,b)
          //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
          //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
          //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
          //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
          // 2) each permutationally-unique integral (shell set) must be scaled by its degeneracy,
          //    i.e. the number of the integrals/sets equivalent to it
          // 3) the end result must be symmetrized
          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;

                  const auto value = buf_1234[f1234];

                  const auto value_scal_by_deg = value * s1234_deg;
                  G(bf1,bf2) += D(bf3,bf4) * value_scal_by_deg;
                  G(bf3,bf4) += D(bf1,bf2) * value_scal_by_deg;
                  G(bf1,bf3) -= 0.25 * D(bf2,bf4) * value_scal_by_deg;
                  G(bf2,bf4) -= 0.25 * D(bf1,bf3) * value_scal_by_deg;
                  G(bf1,bf4) -= 0.25 * D(bf2,bf3) * value_scal_by_deg;
                  G(bf2,bf3) -= 0.25 * D(bf1,bf4) * value_scal_by_deg;
                }
              }
            }
          }
        }
      }
    }
  }

  // symmetrize the result and return
  Matrix Gt = G.transpose();
  return 0.5 * (G + Gt);
}

btas::Tensor<double>
make_F_mo(const Matrix& C, const btas::Tensor<double>& F) {
    btas::Tensor<double> C_ten(C.rows(), C.cols());
    for (int p = 0; p != C.rows(); ++p) {
        for (int q = 0; q != C.cols(); ++q) {
            C_ten(p, q) = C(p, q);
        }
    }

    btas::Tensor<double> F_ma;
    btas::contract(1.0, F, {1, 2}, C_ten, {1, 3}, 0.0, F_ma, {3, 2});

    btas::Tensor<double> F_mm;
    btas::contract(1.0, F_ma, {1, 2}, C_ten, {2, 3}, 0.0, F_mm, {1, 3});

    return F_mm;
}

// transforms <mu nu | rho sigma> -> <pq|rs>
btas::Tensor<double>
    make_v_gg_gg(const Matrix& C, const btas::Tensor<double>& v_aa_aa) {

    btas::Tensor<double> C_ten(C.rows(), C.cols());
    for (int p = 0; p != C.rows(); ++p) {
        for (int q = 0; q != C.cols(); ++q) {
            C_ten(p, q) = C(p, q);
        }
    }

    btas::Tensor<double> v_ga_aa;
    btas::contract(1.0, v_aa_aa, {1, 2, 3, 4}, C_ten, {1, 5}, 0.0, v_ga_aa, {5, 2, 3, 4});

    btas::Tensor<double> v_gg_aa;
    btas::contract(1.0, v_ga_aa, {1, 2, 3, 4}, C_ten, {3, 5}, 0.0, v_gg_aa, {1, 2, 5, 4});

    btas::Tensor<double> v_gg_ga;
    btas::contract(1.0, v_gg_aa, {1, 2, 3, 4}, C_ten, {4, 5}, 0.0, v_gg_ga, {1, 2, 3, 5});

    btas::Tensor<double> v_gg_gg;
    btas::contract(1.0, v_gg_ga, {1, 2, 3, 4}, C_ten, {2, 5}, 0.0, v_gg_gg, {1, 5, 3, 4});

    return v_gg_gg;
}

double mp2_energy(const btas::Tensor<double>& ia_jb, int nocc, int n, const Eigen::VectorXd& evals) {
  double mp2e = 0.0;

  const auto nuocc = n - nocc;
  for (int i = 0; i != nocc; ++i) {
    for (int j = 0; j != nocc; ++j) {
      for (int a = 0; a != nuocc; ++a) {
        for (int b = 0; b != nuocc; ++b) {
          mp2e += ia_jb(i, a, j, b) *
                  (2 * ia_jb(i, a, j, b) - ia_jb(i, b, j, a)) /
                  (evals(i) + evals(j) - evals(nocc+a) - evals(nocc+b));
        }
      }
    }
  }
  return mp2e;
}

btas::Tensor<double> make_v_aa_aa(const libint2::BasisSet& obs)
{
    using libint2::Shell;
    using libint2::Engine;
    using libint2::Operator;

    size_t n = nbasis(obs);
    btas::Tensor<double> rei_ao_ints(n, n, n, n);

    libint2::initialize();

    // construct the electron repulsion integrals engine
    Engine engine(Operator::coulomb, max_nprim(obs), max_nprim(obs), 0);

    auto shell2bf = obs.shell2bf();

    // buf[0] points to the target shell set after every call  to engine.compute()
    const auto& buf = engine.results();

    // loop over shell pairs of the Fock matrix, {s1,s2}
    // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
    for(auto s1=0; s1!=obs.size(); ++s1) {

        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = obs[s1].size();

        for(auto s2=0; s2!=obs.size(); ++s2) {

            auto bf2_first = shell2bf[s2];
            auto n2 = obs[s2].size();

            // loop over shell pairs of the density matrix, {s3,s4}
            // again symmetry is not used for simplicity
            for(auto s3=0; s3!=obs.size(); ++s3) {

                auto bf3_first = shell2bf[s3];
                auto n3 = obs[s3].size();

                for(auto s4=0; s4!= obs.size(); ++s4) {

                    auto bf4_first = shell2bf[s4];
                    auto n4 = obs[s4].size();

                    // Computes (s1 s2 | s3 s4)
                    engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
                    const auto* buf_1234 = buf[0];
                    if (buf_1234 == nullptr)
                        continue; // if all integrals screened out, skip to next quartet

                    // we don't have an analog of Eigen for tensors (yet ... see github.com/BTAS/BTAS, under development)
                    // hence some manual labor here:
                    // 1) loop over every integral in the shell set (= nested loops over basis functions in each shell)
                    // and 2) add contribution from each integral
                    for(auto f1=0, f1234=0; f1!=n1; ++f1) {
                        const auto bf1 = f1 + bf1_first;
                        for(auto f2=0; f2!=n2; ++f2) {
                            const auto bf2 = f2 + bf2_first;
                            for(auto f3=0; f3!=n3; ++f3) {
                                const auto bf3 = f3 + bf3_first;
                                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                                    const auto bf4 = f4 + bf4_first;
                                    rei_ao_ints(bf1, bf3, bf2, bf4) = buf_1234[f1234]; // switched s2 and s3 to get physisist's notation
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return rei_ao_ints;
}

//    // iq_rs(i,q,r,s) = 1.0 * \sum_p pq_rs(p,q,r,s) * C_occ(p,i) + 0.0 * iq_rs(i,q,r,s)
//    btas::contract(1.0, pq_rs, {1, 2, 3, 4}, C_occ, {1, 5}, 0.0, iq_rs, {5, 2, 3, 4});
btas::Tensor<double> make_I_u_u(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> I_u_u; // (nuocc, nuocc)

    btas::Tensor<double> v_temp(nuocc, nuocc, nocc, nocc);
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    v_temp(a, b, i, j) = (-2 * v_uu_oo(a, b, i, j)) + v_uu_oo(b, a, i, j);
                }
            }
        }
    }
    btas::contract(1.0, v_temp, {1, 2, 3, 4}, t_oo_uu, {3, 4, 1, 5}, 0.0, I_u_u, {2, 5});
    return I_u_u;
}

btas::Tensor<double> make_I_o_o(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> I_o_o;
    btas::Tensor<double> v_temp(nuocc, nuocc, nocc, nocc);
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    v_temp(a, b, i, j) = (2 * v_uu_oo(a, b, i, j)) - v_uu_oo(a, b, j, i);
                }
            }
        }
    }
    btas::contract(1.0, v_temp, {1, 2, 3, 4}, t_oo_uu, {3, 5, 1, 2}, 0.0, I_o_o, {5, 4});
    return I_o_o;
}

btas::Tensor<double> make_I_oo_oo(const btas::Tensor<double>& v_oo_oo, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> I_oo_oo(nocc, nocc, nocc, nocc);
    btas::Tensor<double> vt;
    btas::contract(1.0, v_uu_oo, {1, 2, 3, 4}, t_oo_uu, {5, 6, 1, 2}, 0.0, vt, {5, 6, 3, 4});
    I_oo_oo = v_oo_oo + vt;
    return I_oo_oo;
}

btas::Tensor<double> make_I_ou_ou(const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& t_oo_uu, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> I_ou_ou(nocc, nuocc, nocc, nuocc);
    btas::Tensor<double> vt(nocc, nuocc, nocc, nuocc);
    btas::contract(0.5, v_uu_oo, {1, 2, 3, 4}, t_oo_uu, {5, 4, 1, 6}, 0.0, vt, {5, 2, 3, 6});
    I_ou_ou = v_ou_ou - vt;
    return I_ou_ou;
}

btas::Tensor<double> make_I_uo_ou(const btas::Tensor<double> &v_uo_ou, const btas::Tensor<double> &v_uu_oo,
                                  const btas::Tensor<double> &v_ou_ou, const btas::Tensor<double> &t_oo_uu,
                                  const int nocc, const int n)
                                   {
    int nuocc = n - nocc;
    btas::Tensor<double> I_uo_ou;
    btas::Tensor<double> vt_2ndterm;
    btas::contract(0.5, v_uu_oo, {'e', 'b', 'i', 'm'}, t_oo_uu, {'j', 'm', 'a', 'e'}, 0.0, vt_2ndterm, {'b', 'j', 'i', 'a'});
    btas::Tensor<double> t_temp(nocc, nocc, nuocc, nuocc);
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    t_temp(i, j, a, b) = t_oo_uu(i, j, a, b) - (0.5 * t_oo_uu(i, j, b, a));
                }
            }
        }
    }

    btas::Tensor<double> vt_1stterm;
    btas::contract(1.0, v_uu_oo, {'b', 'e', 'i', 'm'}, t_temp, {'m', 'j', 'e', 'a'}, 0.0, vt_1stterm, {'b', 'j', 'i', 'a'});
    I_uo_ou = v_uo_ou + vt_1stterm - vt_2ndterm;
    return I_uo_ou;
}

//R = calcResidual(v_oo_uu, v_uu_uu, t_oo_uu, I_u_u, I_o_o, I_oo_oo, I_ou_ou, I_uo_ou, n, ndocc)
btas::Tensor<double>
        calcResidual(const btas::Tensor<double>& v_oo_uu, const btas::Tensor<double>& v_uu_uu, const btas::Tensor<double>& t,
                     const btas::Tensor<double>& I_u_u, const btas::Tensor<double>& I_o_o, const btas::Tensor<double>& I_oo_oo,
                     const btas::Tensor<double>& I_ou_ou, const btas::Tensor<double>& I_uo_ou, const int n, const int nocc) {

    int nuocc = n - nocc;
    btas::Tensor<double> R(nocc, nocc, nuocc, nuocc);

    btas::Tensor<double> t_oo_uu_I_u_u; // first term in permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_u_u, {4, 5}, 0.0, t_oo_uu_I_u_u, {1, 2, 3, 5});

    btas::Tensor<double> t_oo_uu_I_o_o; // second term in permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_o_o, {5, 2}, 0.0, t_oo_uu_I_o_o, {1, 5, 3, 4});

    btas::Tensor<double> v_uu_uu_t_oo_uu; // third term in permutative equation, note already scaled
    btas::contract(0.5, v_uu_uu, {1, 2, 3, 4}, t, {5, 6, 1, 2}, 0.0, v_uu_uu_t_oo_uu, {5, 6, 3, 4});

    btas::Tensor<double> t_oo_uu_I_oo_oo; // fourth term in permutative equation, note already scaled
    btas::contract(0.5, t, {1, 2, 3, 4}, I_oo_oo, {5, 6, 1, 2}, 0.0, t_oo_uu_I_oo_oo, {5, 6, 3, 4});

    btas::Tensor<double> t_oo_uu_I_ou_ou; // fifth term in permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_ou_ou, {5, 4, 1, 6}, 0.0, t_oo_uu_I_ou_ou, {5, 2, 3, 6});

    btas::Tensor<double> I_ou_ou_t_oo_uu; // sixth term in permutative equation
    btas::contract(1.0, I_ou_ou, {1, 2, 3, 4}, t, {3, 5, 2, 6}, 0.0, I_ou_ou_t_oo_uu, {1, 5, 4, 6});

    btas::Tensor<double> t_temp(nocc, nocc, nuocc, nuocc);
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    t_temp(i, j, a, b) = (2 * t(i, j, a, b)) - t(j, i, a, b);
                }
            }
        }
    }

    btas::Tensor<double> t_oo_uu_I_uo_ou; // seventh term in permutative equation
    btas::contract(1.0, t_temp, {1, 2, 3, 4}, I_uo_ou, {3, 5, 1, 6}, 0.0, t_oo_uu_I_uo_ou, {2, 5, 4, 6});

    t_oo_uu_I_u_u = ccd_permute(t_oo_uu_I_u_u);
    t_oo_uu_I_o_o = ccd_permute(t_oo_uu_I_o_o);
    v_uu_uu_t_oo_uu = ccd_permute(v_uu_uu_t_oo_uu);
    t_oo_uu_I_oo_oo = ccd_permute(t_oo_uu_I_oo_oo);
    t_oo_uu_I_ou_ou = ccd_permute(t_oo_uu_I_ou_ou);
    I_ou_ou_t_oo_uu = ccd_permute(I_ou_ou_t_oo_uu);
    t_oo_uu_I_uo_ou = ccd_permute(t_oo_uu_I_uo_ou);
    R = v_oo_uu + t_oo_uu_I_u_u - t_oo_uu_I_o_o + v_uu_uu_t_oo_uu + t_oo_uu_I_oo_oo - t_oo_uu_I_ou_ou - I_ou_ou_t_oo_uu + t_oo_uu_I_uo_ou;
    return R;
}

double calc_ccd_energy(btas::Tensor<double> const & t_oo_uu, btas::Tensor<double> const & v_oo_uu, int n, int nocc) {
    double ccd_e = 0.0;
    auto nuocc = n - nocc;

    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    double change = (2 * v_oo_uu(i, j, a, b) - v_oo_uu(i, j, b, a)) * t_oo_uu(i, j, a, b);
                    ccd_e += change;
                }
            }
        }
    }
    return ccd_e;
}

btas::Tensor<double> make_h_j_i(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau) {
    btas::Tensor<double> v_temp(v_uu_oo.extent(0), v_uu_oo.extent(1), v_uu_oo.extent(2), v_uu_oo.extent(3));
    for (size_t a = 0; a != v_temp.extent(0); ++a) {
        for (size_t b = 0; b != v_temp.extent(1); ++b) {
            for (size_t i = 0; i != v_temp.extent(2); ++i) {
                for (size_t j = 0; j != v_temp.extent(3); ++j) {
                    v_temp(a,b,i,j) = 2 * v_uu_oo(a,b,i,j) + v_uu_oo(b,a,i,j);
                }
            }
        }
    }

    btas::Tensor<double> vTau;
    btas::contract(1.0, v_temp, {'a', 'b', 'i', 'j'}, tau, {'u', 'j', 'a', 'b'}, 0.0, vTau, {'u', 'i'});

    return fock_matrix + vTau;
}

btas::Tensor<double> make_h_b_a(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau) {
    btas::Tensor<double> v_temp(v_uu_oo.extent(0), v_uu_oo.extent(1), v_uu_oo.extent(2), v_uu_oo.extent(3));
    for (size_t a = 0; a != v_temp.extent(0); ++a) {
        for (size_t b = 0; b != v_temp.extent(1); ++b) {
            for (size_t i = 0; i != v_temp.extent(2); ++i) {
                for (size_t j = 0; j != v_temp.extent(3); ++j) {
                    v_temp(a,b,i,j) = 2 * v_uu_oo(a,b,i,j) + v_uu_oo(a,b,j,i);
                }
            }
        }
    }

    btas::Tensor<double> vTau;
    btas::contract(1.0, v_temp, {'a', 'b', 'i', 'j'}, tau, {'i', 'j', 'B', 'b'}, 0.0, vTau, {'a', 'B'});

    return fock_matrix - vTau;
}

btas::Tensor<double> make_h_b_i(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& singles_amps) {
    btas::Tensor<double> v_temp(v_uu_oo.extent(0), v_uu_oo.extent(1), v_uu_oo.extent(2), v_uu_oo.extent(3));
    for (size_t a = 0; a != v_temp.extent(0); ++a) {
        for (size_t b = 0; b != v_temp.extent(1); ++b) {
            for (size_t i = 0; i != v_temp.extent(2); ++i) {
                for (size_t j = 0; j != v_temp.extent(3); ++j) {
                    v_temp(a,b,i,j) = 2 * v_uu_oo(a,b,i,j) + v_uu_oo(b,a,i,j);
                }
            }
        }
    }

    btas::Tensor<double> vt;
    btas::contract(1.0, v_temp, {'a', 'b', 'i', 'j'}, vt, {'j', 'b'}, 0.0, vt, {'a', 'i'});

    return fock_matrix + vt;
}

btas::Tensor<double> make_g_j_i(const btas::Tensor<double>& h_j_i, const btas::Tensor<double>& fock_matrix,
                                const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_ou_oo) {
    btas::Tensor<double> ft;
    btas::contract(1.0, fock_matrix, {'a', 'i'}, singles_amps, {'u', 'a'}, 0.0, ft, {'u', 'i'});

    btas::Tensor<double> v_temp(v_ou_oo.extent(0), v_ou_oo.extent(1), v_ou_oo.extent(2), v_ou_oo.extent(3));
    for (size_t u = 0; u != v_ou_oo.extent(0); ++u) {
        for (size_t a = 0; a != v_ou_oo.extent(1); ++a) {
            for (size_t i = 0; i != v_ou_oo.extent(2); ++i) {
                for (size_t j = 0; j != v_ou_oo.extent(3); ++j) {
                    v_temp(u,a,i,j) = 2 * v_ou_oo(u,a,i,j) + v_ou_oo(u,a,j,i);
                }
            }
        }
    }
    btas::Tensor<double> vt;
    btas::contract(1.0, v_temp, {'u', 'a', 'i', 'j'}, singles_amps, {'j', 'a'}, 0.0, vt, {'u', 'i'});

    return h_j_i + ft + vt;
}

btas::Tensor<double> make_g_b_a(const btas::Tensor<double>& h_b_a, const btas::Tensor<double>& fock_matrix,
                                const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_uo) {
    btas::Tensor<double> ft;
    btas::contract(1.0, fock_matrix, {'a', 'i'}, singles_amps, {'i', 'B'}, 0.0, ft, {'a', 'b'});

    btas::Tensor<double> v_temp(v_uu_uo.extent(0), v_uu_uo.extent(1), v_uu_uo.extent(2), v_uu_uo.extent(3));
    for (size_t a = 0; a != v_uu_uo.extent(0); ++a) {
        for (size_t b = 0; b != v_uu_uo.extent(1); ++b) {
            for (size_t B = 0; B != v_uu_uo.extent(2); ++B) {
                for (size_t i = 0; i != v_uu_uo.extent(3); ++i) {
                    v_temp(a,b,B,i) = 2 * v_uu_uo(a,b,B,i) + v_uu_uo(b,a,B,i);
                }
            }
        }
    }
    btas::Tensor<double> vt;
    btas::contract(1.0, v_temp, {'a', 'b', 'B', 'i'}, singles_amps, {'i', 'b'}, 0.0, vt, {'a', 'B'});

    return h_b_a - ft + vt;
}

btas::Tensor<double> make_a_uv_ij(const btas::Tensor<double>& v_oo_oo, const btas::Tensor<double>& v_ou_oo,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uo_oo,
                                  const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau) {
    btas::Tensor<double> vt1;
    btas::contract(1.0, v_ou_oo, {'u', 'a', 'i', 'j'}, singles_amps, {'v', 'a'}, 0.0, vt1, {'u', 'v', 'i', 'j'});

    btas::Tensor<double> vt2;
    btas::contract(1.0, v_uo_oo, {'a', 'v', 'i', 'j'}, singles_amps, {'u', 'a'}, 0.0, vt2, {'u', 'v', 'i', 'j'});

    btas::Tensor<double> vTau;
    btas::contract(1.0, v_uu_oo, {'a', 'b', 'i', 'j'}, tau, {'u', 'v', 'a', 'b'}, 0.0, vTau, {'u', 'v', 'i', 'j'});

    return v_oo_oo + vt1 + vt2 + vTau;
}

btas::Tensor<double> make_b_ab_BY(const btas::Tensor<double>& v_uu_uu, const btas::Tensor<double>& v_uu_uo,
                                      const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_ou) {
    btas::Tensor<double> vt1;
    btas::contract(1.0, v_uu_uo, {'a', 'b', 'B', 'i'}, singles_amps, {'i', 'Y'}, 0.0, vt1, {'a', 'b', 'B', 'Y'});

    btas::Tensor<double> vt2;
    btas::contract(1.0, v_uu_ou, {'a', 'b', 'i', 'Y'}, singles_amps, {'i', 'B'}, 0.0, vt2, {'a', 'b', 'B', 'Y'});

    return v_uu_uu + vt1 + vt2;
}

btas::Tensor<double> make_j_ua_Bi(const btas::Tensor<double>& v_ou_uo, const btas::Tensor<double>& v_ou_oo,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_uo,
                                  const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& T, const btas::Tensor<double>& doubles_amps) {
    btas::Tensor<double> vt1;
    btas::contract(1.0, v_ou_oo, {'u', 'a', 'j', 'i'}, singles_amps, {'j', 'B'}, 0.0, vt1, {'u', 'a', 'B', 'i'});

    btas::Tensor<double> vt2;
    btas::contract(1.0, v_uu_uo, {'b', 'a', 'B', 'i'}, singles_amps, {'u', 'b'}, 0.0, vt2, {'u', 'a', 'B', 'i'});

    btas::Tensor<double> vT;
    btas::contract(1.0, v_uu_oo, {'a', 'b', 'i', 'j'}, T, {'u', 'j', 'b', 'B'}, 0.0, vT, {'u', 'a', 'B', 'i'});

    btas::Tensor<double> v_temp(v_uu_oo.extent(0), v_uu_oo.extent(1), v_uu_oo.extent(2), v_uu_oo.extent(3));
    for (size_t a = 0; a != v_uu_oo.extent(0); ++a) {
        for (size_t b = 0; b != v_uu_oo.extent(1); ++b) {
            for (size_t i = 0; i != v_uu_oo.extent(2); ++i) {
                for (size_t j = 0; j != v_uu_oo.extent(3); ++j) {
                    v_temp(a,b,i,j) = 2 * v_uu_oo(a,b,i,j) - v_uu_oo(b,a,i,j);
                }
            }
        }
    }

    btas::Tensor<double> vtd;
    btas::contract(1.0, v_temp, {'a', 'b', 'i', 'j'}, doubles_amps, {'u', 'j', 'B', 'b'}, 0.0, vtd, {'u', 'a', 'B', 'i'});

    return v_ou_uo - vt1 + vt2 - vT + vtd;
}

btas::Tensor<double> make_k_ua_iB(const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_ou,
                                  const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& v_uu_oo,
                                  const btas::Tensor<double>& T) {
    btas::Tensor<double> vt1;
    btas::contract(1.0, v_ou_ou, {'u', 'a', 'i', 'j'}, singles_amps, {'j', 'B'}, 0.0, vt1, {'u', 'a', 'i', 'B'});

    btas::Tensor<double> vt2;
    btas::contract(1.0, v_uu_ou, {'b', 'a', 'i', 'B'}, singles_amps, {'u', 'b'}, 0.0, vt2, {'u', 'a', 'i', 'B'});

    btas::Tensor<double> vT;
    btas::contract(1.0, v_uu_oo, {'b', 'a', 'i', 'j'}, T, {'u', 'j', 'b', 'B'}, 0.0, vT, {'u', 'a', 'i', 'B'});

    return v_ou_ou - vt1 + vt2 - vT;
}

btas::Tensor<double> t1_residual(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& singles_amps,
                                 const btas::Tensor<double>& h_b_a, const btas::Tensor<double>& h_j_i, const btas::Tensor<double>& h_b_i,
                                 const btas::Tensor<double>& doubles_amps, const btas::Tensor<double>& v_uo_ou,
                                 const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_ou, const btas::Tensor<double>& tau,
                                 const btas::Tensor<double>& v_uo_oo, const btas::Tensor<double>& v_uu_oo, int nuocc, int nocc) {
    btas::Tensor<double> term2_int;
    btas::contract(1.0, fock_matrix, {'i', 'a'}, singles_amps, {'i', 'b'}, 0.0, term2_int, {'a', 'b'});
    btas::Tensor<double> term2;
    btas::contract(1.0, term2_int, {'a', 'b'}, singles_amps, {'i', 'a'}, 0.0, term2, {'i', 'b'});

    btas::Tensor<double> term3;
    btas::contract(1.0, h_b_a, {'b', 'a'}, singles_amps, {'i', 'b'}, 0.0, term3, {'i', 'a'});

    btas::Tensor<double> term4;
    btas::contract(1.0, h_j_i, {'j', 'i'}, singles_amps, {'i', 'a'}, 0.0, term4, {'j', 'a'});

    btas::Tensor<double> singles_cont;
    btas::contract(1.0, singles_amps, {'i', 'a'}, singles_amps, {'j', 'b'}, 0.0, singles_cont, {'i', 'j', 'a', 'b'});
    btas::Tensor<double> term5_int(doubles_amps.extent(0), doubles_amps.extent(1), doubles_amps.extent(2), doubles_amps.extent(3));
    for (size_t i = 0; i != doubles_amps.extent(0); ++i) {
        for (size_t j = 0; j != doubles_amps.extent(1); ++j) {
            for (size_t a = 0; a != doubles_amps.extent(2); ++a) {
                for (size_t b = 0; b != doubles_amps.extent(3); ++b) {
                    term5_int(i, j, a, b) = 2 * doubles_amps(i, j, a, b) + doubles_amps(j, i, a, b) + singles_cont(i, j, a, b);
                }
            }
        }
    }
    btas::Tensor<double> term5;
    btas::contract(1.0, h_b_i, {'b', 'i'}, term5_int, {'i', 'j', 'a', 'b'}, 0.0, term5, {'j', 'b'});

    /// TODO: This is a problem...
    btas::Tensor<double> term6_int(nuocc, nocc, nocc, nuocc);
    for (size_t a = 0; a != nuocc; ++a) {
        for (size_t i = 0; i != nocc; ++i) {
            for (size_t j = 0; j != nocc; ++j) {
                for (size_t b = 0; b != nuocc; ++b) {
                    term6_int(a, i, j, b) = 2 * v_uo_ou(a, i, j, b) - v_ou_ou(i, a, j, b);
                }
            }
        }
    }
    btas::Tensor<double> term6;
    btas::contract(1.0, term6_int, {'a', 'i', 'j', 'b'}, singles_amps, {'i', 'a'}, 0.0, term6, {'j', 'b'});

    btas::Tensor<double> term7_int(v_uu_ou.extent(0), v_uu_ou.extent(1), v_uu_ou.extent(2), v_uu_ou.extent(3));
    for (size_t a = 0; a != v_uu_ou.extent(0); ++a) {
        for (size_t b = 0; b != v_uu_ou.extent(1); ++b) {
            for (size_t i = 0; i != v_uu_ou.extent(2); ++i) {
                for (size_t c = 0; c != v_uu_ou.extent(3); ++c) {
                    term7_int(a, b, i, c) = 2 * v_uu_ou(a, b, i, c) - v_uu_ou(b, a, i, c);
                }
            }
        }
    }
    btas::Tensor<double> term7;
    btas::contract(1.0, term7_int, {'a', 'b', 'i', 'c'}, tau, {'i', 'j', 'a', 'b'}, 0.0, term7, {'j', 'c'});

    btas::Tensor<double> term8_int(v_uo_oo.extent(0), v_uo_oo.extent(1), v_uo_oo.extent(2), v_uo_oo.extent(3));
    for (size_t a = 0; a != v_uo_oo.extent(0); ++a) {
        for (size_t i = 0; i != v_uo_oo.extent(1); ++i) {
            for (size_t j = 0; j != v_uo_oo.extent(2); ++j) {
                for (size_t k = 0; k != v_uo_oo.extent(3); ++k) {
                    term8_int(a, i, j, k) = 2 * v_uo_oo(a, i, j, k) - v_uo_oo(a, i, k, j);
                }
            }
        }
    }
    btas::Tensor<double> term8;
    btas::contract(1.0, term8_int, {'a', 'i', 'j', 'k'}, tau, {'j', 'k', 'a', 'b'}, 0.0, term8, {'i', 'b'});

    return fock_matrix - term2 + term3 - term4 + term5 + term6 + term7 - term8;
}

btas::Tensor<double> t2_residual(const btas::Tensor<double>& v_oo_uu, const btas::Tensor<double>& a_oo_oo,
                                 const btas::Tensor<double>& tau, const btas::Tensor<double>& b_uu_uu, const btas::Tensor<double>& g_u_u,
                                 const btas::Tensor<double>& singles_amps, const btas::Tensor<double>& g_o_o,
                                 const btas::Tensor<double>& v_ou_uu, const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_oo_uo,
                                 const btas::Tensor<double>& v_ou_uo, const btas::Tensor<double>& j_ou_uo, const btas::Tensor<double>& k_ou_ou,
                                 const btas::Tensor<double>& doubles_amps, int nuocc, int nocc) {
    btas::Tensor<double> term2;
    btas::contract(1.0, a_oo_oo, {'u', 'v', 'i', 'j'}, tau, {'i', 'j', 'B', 'Y'}, 0.0, term2, {'u', 'v', 'B', 'Y'});

    btas::Tensor<double> term3;
    btas::contract(1.0, b_uu_uu, {'a', 'b', 'B', 'Y'}, tau, {'u', 'v', 'a', 'b'}, 0.0, term3, {'u', 'v', 'B', 'Y'});

    btas::Tensor<double> term4;
    btas::contract(1.0, g_u_u, {'a', 'B'}, doubles_amps, {'u', 'v', 'a', 'Y'}, 0.0, term4, {'u', 'v', 'B', 'Y'});
    term4 = ccd_permute(term4);

    btas::Tensor<double> term5;
    btas::contract(1.0, g_o_o, {'u', 'i'}, doubles_amps, {'i', 'v', 'B', 'Y'}, 0.0, term5, {'u', 'v', 'B', 'Y'});
    term5 = ccd_permute(term5);

    btas::Tensor<double> term6_int1;
    btas::contract(1.0, v_ou_ou, {'u', 'a', 'i', 'Y'}, singles_amps, {'i', 'B'}, 0.0, term6_int1, {'u', 'a', 'B', 'Y'});
    btas::Tensor<double> term6_int2(v_ou_uu.extent(0), v_ou_uu.extent(1), v_ou_uu.extent(2), v_ou_uu.extent(3));
    for (size_t i = 0; i != v_ou_uu.extent(0); ++i) {
        for (size_t a = 0; a != v_ou_uu.extent(1); ++a) {
            for (size_t b = 0; b != v_ou_uu.extent(2); ++b) {
                for (size_t c = 0; c != v_ou_uu.extent(3); ++c) {
                    term6_int2(i, a, b, c) = v_ou_uu(i, a, b, c) - term6_int1(i, a, b, c);
                }
            }
        }
    }
    btas::Tensor<double> term6;
    btas::contract(1.0, term6_int2, {'u', 'a', 'B', 'Y'}, singles_amps, {'v', 'a'}, 0.0, term6, {'u', 'v', 'B', 'Y'});
    term6 = ccd_permute(term6);

    btas::Tensor<double> term7_int1;
    btas::contract(1.0, v_ou_uo, {'u', 'a', 'B', 'i'}, singles_amps, {'v', 'a'}, 0.0, term7_int1, {'u', 'v', 'B', 'i'});
    btas::Tensor<double> term7_int2(v_oo_uo.extent(0), v_oo_uo.extent(1), v_oo_uo.extent(2), v_oo_uo.extent(3));
    for (size_t i = 0; i != v_oo_uo.extent(0); ++i) {
        for (size_t j = 0; j != v_oo_uo.extent(1); ++j) {
            for (size_t a = 0; a != v_oo_uo.extent(2); ++a) {
                for (size_t k = 0; k != v_oo_uo.extent(3); ++k) {
                    term7_int2(i, j, a, k) = v_oo_uo(i, j, a, k) - term6_int1(i, j, a, k);
                }
            }
        }
    }
    btas::Tensor<double> term7;
    btas::contract(1.0, term7_int2, {'u', 'v', 'B', 'i'}, singles_amps, {'i', 'Y'}, 0.0, term7, {'u', 'v', 'B', 'Y'});
    term7 = ccd_permute(term7);

    btas::Tensor<double> term8_int1(nocc, nuocc, nuocc, nocc); //TODO: This is also a problem
    for (size_t i = 0; i != nocc; ++i) {
        for (size_t a = 0; a != nuocc; ++a) {
            for (size_t b = 0; b != nuocc; ++b) {
                for (size_t j = 0; j != nocc; ++j) {
                    term8_int1(i, a, b, j) = 2 * j_ou_uo(i, a, b, j) - k_ou_ou(i, a, j, b);
                }
            }
        }
    }
    btas::Tensor<double> term8_int2(nocc, nocc, nuocc, nuocc);
    for (size_t i = 0; i != nocc; ++i) {
        for (size_t j = 0; j != nocc; ++j) {
            for (size_t a = 0; a != nuocc; ++a) {
                for (size_t b = 0; b != nuocc; ++b) {
                    term8_int2(i, j, a, b) = 2 * doubles_amps(i, j, a, b) - doubles_amps(i, j, b, a);
                }
            }
        }
    }
    btas::Tensor<double> term8;
    btas::contract(0.5, term8_int1, {'u', 'a', 'B', 'i'}, term8_int2, {'i', 'v', 'a', 'Y'}, 0.0, term8, {'u', 'v', 'B', 'Y'});
    term8 = ccd_permute(term8);

    btas::Tensor<double> term9;
    btas::contract(0.5, k_ou_ou, {'u', 'a', 'i', 'B'}, doubles_amps, {'i', 'v', 'Y', 'a'}, 0.0, term9, {'u', 'v', 'B', 'Y'});
    term9 = ccd_permute(term9);

    btas::Tensor<double> term10;
    btas::contract(1.0, k_ou_ou, {'u', 'a', 'i', 'Y'}, doubles_amps, {'i', 'v', 'B', 'a'}, 0.0, term10, {'u', 'v', 'B', 'Y'});
    term10 = ccd_permute(term10);

    return v_oo_uu + term2 + term3 + term4 - term5 + term6 - term7 + term8 - term9 - term10;
}

btas::Tensor<double> ccd_permute(btas::Tensor<double>& tensor) {
    btas::Tensor<double> permuted(tensor.extent(0), tensor.extent(1), tensor.extent(2), tensor.extent(3));
    for (int i = 0; i != tensor.extent(0); ++i) {
        for (int j = 0; j != tensor.extent(1); ++j) {
            for (int a = 0; a != tensor.extent(2); ++a) {
                for (int b = 0; b != tensor.extent(3); ++b) {
                    auto v = tensor(i, j, a, b) + tensor(j, i, b, a);
                    permuted(i,j,a,b) = v;
                    permuted(j,i,b,a) = v;
                }
            }
        }
    }
    return permuted;
}

double calc_ccsd_energy(const btas::Tensor<double>& fock_matrix, const btas::Tensor<double>& singles_amplitudes,
                        const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& tau) {
    btas::Tensor<double> ft;
    btas::contract(2.0, fock_matrix, {'a', 'i'}, singles_amplitudes, {'i', 'a'}, 0.0, ft, {'e'}); /// Problem?

    btas::Tensor<double> v_temp(v_uu_oo.extent(0), v_uu_oo.extent(1), v_uu_oo.extent(2), v_uu_oo.extent(3));
    for (size_t a = 0; a != v_uu_oo.extent(0); ++a) {
        for (size_t b = 0; b != v_uu_oo.extent(1); ++b) {
            for (size_t i = 0; i != v_uu_oo.extent(2); ++i) {
                for (size_t j = 0; j != v_uu_oo.extent(3); ++j) {
                    v_temp(a,b,i,j) = 2 * v_uu_oo(a,b,i,j) - v_uu_oo(b,a,i,j);
                }
            }
        }
    }

    btas::Tensor<double> vTau;
    btas::contract(1.0, v_temp, {'a', 'b', 'i', 'j'}, tau, {'i', 'j', 'a', 'b'}, 0.0, vTau, {'e'});

    return ft(0,0) + vTau(0,0,0,0); // TODO: Clean up this return statement, data types? Extents of the tensors?
}

btas::Tensor<double> make_tau(const btas::Tensor<double>& doubles_amps, const btas::Tensor<double>& singles_amps) {
    btas::Tensor<double> singles_product;
    btas::contract(1.0, singles_amps, {'i', 'a'}, singles_amps, {'j', 'b'}, 0.0, singles_product, {'i', 'j', 'a', 'b'});

    return doubles_amps + singles_product;
}

btas::Tensor<double> make_T(const btas::Tensor<double>& doubles_amps, const btas::Tensor<double>& singles_amps) {
    btas::Tensor<double> singles_product;
    btas::contract(1.0, singles_amps, {'i', 'a'}, singles_amps, {'j', 'b'}, 0.0, singles_product, {'i', 'j', 'a', 'b'});

    btas::Tensor<double> doubles_copy = doubles_amps;
    for (int i = 0; i != doubles_amps.extent(0); ++i) {
        for (int j = 0; j != doubles_amps.extent(1); ++j) {
            for (int a = 0; a != doubles_amps.extent(2); ++a) {
                for (int b = 0; b != doubles_amps.extent(3); ++b) {
                    doubles_copy(i,j,a,b) = doubles_copy(i,j,a,b) * 0.5;
                }
            }
        }
    }

    return doubles_copy + singles_product;
}