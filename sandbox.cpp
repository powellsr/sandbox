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
btas::Tensor<double> transform_pqrs_to_iajb(const Matrix& C, const btas::Tensor<double>& pq_rs, int nocc);
btas::Tensor<double> rei_ao_integrals_tensor(libint2::BasisSet& obs);

btas::Tensor<double>
cc_amps_update(const btas::Tensor<double>& v_oo_uu, const btas::Tensor<double>& v_uu_uu, const btas::Tensor<double>& t,
               const btas::Tensor<double>& I_a_b, const btas::Tensor<double>& I_i_j, const btas::Tensor<double>& I_ij_kl,
               const btas::Tensor<double>& I_ia_jb, const btas::Tensor<double>& I_ia_bj, const int n, const int nocc);
btas::Tensor<double> make_density_tensor(const Matrix& fock_matrix, const int nocc, const int n);
btas::Tensor<double> make_I_bj_ia(const btas::Tensor<double>& v_uo_ou, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n);
btas::Tensor<double> make_I_jb_ia(const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n);
btas::Tensor<double> make_I_kl_ij(const btas::Tensor<double>& v_oo_oo, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n);
btas::Tensor<double> make_I_j_i(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n);
btas::Tensor<double> make_I_b_a(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n);
double calc_ccd_energy(btas::Tensor<double>& t, btas::Tensor<double> v, int n, int nocc);

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
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    std::vector<libint2::Atom> atoms = read_geometry(filename);
	const auto basisname = (argc > 2) ? argv[2] : "sto-3g"; //aug-cc-pVDZ

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
    const auto conv = 1e-12;
    auto iter = 0;
    auto rmsd = 0.0;
    auto ediff = 0.0;
    auto ehf = 0.0;
    auto emp2 = 0.0; // MP2 energy
    Matrix F;
    do {
      const auto tstart = std::chrono::high_resolution_clock::now();
      ++iter;

      // Save a copy of the energy and the density
      auto ehf_last = ehf;
      auto D_last = D;

      // build a new Fock matrix
      F = H;//auto //TODO: F scoping
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

    // ####### Begin MP2 stuff
    //std::vector<double> ao_ints = ao_integrals_vector(obs);
    //std::cout << ao_ints << "\n";

    //std::cout << "Length of ao_ints vector: " << ao_ints.size() << "\n";
    //double total_vector_sum = 0.0;
    double total_tensor_sum = 0.0;

    btas::Tensor<double> v_ao = rei_ao_integrals_tensor(obs);
    auto ia_jb = transform_pqrs_to_iajb(C, v_ao, ndocc);
    auto mp2_e_ten = mp2_energy(ia_jb, ndocc, nao, eps);
    std::cout << "Finished MP2\n";

    auto nuocc = obs.size() - ndocc;

    // CCD loop
    btas::Tensor<double> v_oo_oo(ndocc, ndocc, ndocc, ndocc);
    for (int i = 0; i != ndocc; ++i) {
        for (int j = 0; j != ndocc; ++j) {
            for (int k = 0; k != ndocc; ++k) {
                for (int l = 0; l != ndocc; ++l) {
                    v_oo_oo(i, j, k, l) = v_ao(i, j, k, l);
                }
            }
        }
    }

    btas::Tensor<double> v_uu_uu(nuocc, nuocc, nuocc, nuocc);
    for (int a = 0; a != nuocc; ++a) {
        for (int b = 0; b != nuocc; ++b) {
            for (int c = 0; c != nuocc; ++c) {
                for (int d = 0; d != nuocc; ++d) {
                    v_uu_uu(a, b, c, d) = v_ao(a + ndocc, b + ndocc, c + ndocc, d + ndocc);
                }
            }
        }
    }

    btas::Tensor<double> v_uu_oo(nuocc, nuocc, ndocc, ndocc);
    for (int a = 0; a != nuocc; ++a) {
        for (int b = 0; b != nuocc; ++b) {
            for (int i = 0; i != ndocc; ++i) {
                for (int j = 0; j != ndocc; ++j) {
                    v_uu_oo(a, b, i, j) = v_ao(a + ndocc, b + ndocc, i, j);
                }
            }
        }
    }

    btas::Tensor<double> v_oo_uu(ndocc, ndocc, nuocc, nuocc);
    for (int i = 0; i != ndocc; ++i) {
        for (int j = 0; j != ndocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                   v_oo_uu(i, j, a, b) = v_ao(i, j, a + ndocc, b + ndocc);
                }
            }
        }
    }

    btas::Tensor<double> v_ou_ou(ndocc, nuocc, ndocc, nuocc);
    for (int i = 0; i != ndocc; ++i) {
        for (int a = 0; a != nuocc; ++a) {
            for (int j = 0; j != nuocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    v_ou_ou(i, a, j, b) = v_ao(i, a + ndocc, j, b + ndocc);
                }
            }
        }
    }

    btas::Tensor<double> v_uo_ou(nuocc, ndocc, ndocc, nuocc);
    for (int a = 0; a != nuocc; ++a) {
        for (int i = 0; i != ndocc; ++i) {
            for (int j = 0; j != nuocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    v_ou_ou(a, i, j, b) = v_ao(i, a + ndocc, j, b + ndocc);
                }
            }
        }
    }

    const int max_cc_iter = 1;
    int cc_iter = 0;
    //int nuocc = nao - ndocc;
    btas::Tensor<double> t_ij_ab(ndocc, ndocc, nuocc, nuocc); // dimensions?
    t_ij_ab.fill(0);
    btas::Tensor<double> D_tensor(ndocc, ndocc, nuocc, nuocc);
    btas::Tensor<double> I_bj_ia(ndocc, nuocc, ndocc, nuocc);
    btas::Tensor<double> I_jb_ia(ndocc, nuocc, ndocc, nuocc);
    btas::Tensor<double> I_kl_ij(ndocc, nuocc, ndocc, nuocc);
    btas::Tensor<double> I_j_i(ndocc, ndocc);
    btas::Tensor<double> I_b_a(nuocc, nuocc);
    btas::Tensor<double> Dt;

    btas::Range init({btas::Range1{ndocc}, btas::Range1{nuocc}, btas::Range1{ndocc}, btas::Range1(nuocc)});
    ////std::initializer_list<btas::Range1d<
    //std::initializer_list<btas::Range1d<btas::Range1d<int>> init;
    //auto v = rei_ao_ints_ten.slice<std::initializer_list<btas::Range1d<long>>>({btas::Range1d{ndocc}, btas::Range1d{nuocc}, btas::Range1d{ndocc}, btas::Range1d{nuocc}});
    //auto v = rei_ao_ints_ten.slice({btas::Range1{ndocc}, btas::Range1{nuocc}, btas::Range1{ndocc}, btas::Range1(nuocc)});

    do {
        printf("%2i, %2i, %2i, %2i\n", v_ao.extent(0), v_ao.extent(1), v_ao.extent(2), v_ao.extent(3));
        ++cc_iter;
        std::cout << "Entered CCD loop\n";
        D_tensor = make_density_tensor(F, ndocc, nao);
        std::cout << "Made D\n";
        I_bj_ia = make_I_bj_ia(v_uo_ou, v_uu_oo, t_ij_ab, ndocc, nao);
        std::cout << "Made D and I_ia_bj\n";
        I_jb_ia = make_I_jb_ia(v_ou_ou, v_uu_oo, t_ij_ab, ndocc, nao); // working
        std::cout << "Made D and I_ia_jb\n";
        I_kl_ij = make_I_kl_ij(v_oo_oo, v_uu_oo, t_ij_ab, ndocc, nao);
        std::cout << "Made D and I's with four indices\n";
        I_j_i = make_I_j_i(v_uu_oo, t_ij_ab, ndocc, nao);
        I_b_a = make_I_b_a(v_uu_oo, t_ij_ab, ndocc, nao);
        Dt = cc_amps_update(v_oo_uu, v_uu_uu, t_ij_ab, I_b_a, I_j_i, I_kl_ij, I_jb_ia, I_bj_ia, nao, ndocc);
        printf("%2i, %2i, %2i, %2i\n", Dt.extent(0), Dt.extent(1), Dt.extent(2), Dt.extent(3));

        // assuming D(i, a, j, b)
        /*
        auto i_extent = D.extent(0), a_extent = D.extent(1);
        auto ai_extent = a_extent * i_extent;
        btas::Range r_ai_mat({btas::Range1{ai_extent}, btas::Range1{ai_extent}}), r_tensor = D.range();
        D.resize(r_ai_mat); // D(i,a,j,b) -> D(ia,jb)
        Dt.resize(r_ai_mat);
        // Solve Dt = D * t
        auto cholesky = btas::cholesky_inverse(D, Dt);
        if (!cholesky) {
            auto inv = btas::Inverse_Matrix(D);
            if(!inv) {
                bool fast_inv = false;
                btas::pseudoInverse(D, fast_inv);
            }
            btas::contract(1.0, D, {1, 2}, Dt, {2, 3}, 0.0, t, {1, 3});
        }
        t_ij_ab = Dt.resize(r_tensor);
        D.resize(r_tensor);
         */
    } while (cc_iter != max_cc_iter);

    printf("** Vector sum total %20.12f\n", total_tensor_sum);
    printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);
    //printf("** MP2 energy = %20.12f\n", mp2_e);
    printf("** MP2 energy using tensor = %20.12f\n", mp2_e_ten);
    //printf("** Total MP2 energy = %20.12f\n", ehf + enuc + mp2_e);

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

// transforms (pq|rs) -> (ia|jb)
btas::Tensor<double>
    transform_pqrs_to_iajb(const Matrix& C, const btas::Tensor<double>& pq_rs, int nocc) {
    const int n = pq_rs.extent(0);
    const int nuocc = n - nocc;

    printf("nocc= %2i,  n= %2i,  nuocc= %2i\n", nocc, n, nuocc);

    btas::Tensor<double> C_occ(n, nocc);

    for (int mu = 0; mu != n; ++mu) {
        for (int i = 0; i < nocc; ++i) {
            C_occ(mu, i) = C(mu, i);
        }
    }

    btas::Tensor<double> C_uocc(n, nuocc);
    for (int mu = 0; mu != n; ++mu) {
       for (int a = nocc; a != n; ++a) {
            C_uocc(mu, a-nocc) = C(mu, a);
        }
    }

    btas::Tensor<double> iq_rs;  // (iq|rs)
    // iq_rs(i,q,r,s) = 1.0 * \sum_p pq_rs(p,q,r,s) * C_occ(p,i) + 0.0 * iq_rs(i,q,r,s)
    btas::contract(1.0, pq_rs, {1, 2, 3, 4}, C_occ, {1, 5}, 0.0, iq_rs, {5, 2, 3, 4});
    btas::Tensor<double> iq_js;
    btas::contract(1.0, iq_rs, {1, 2, 3, 4}, C_occ, {3, 5}, 0.0, iq_js, {1, 2, 5, 4});
    btas::Tensor<double> iq_jb;
    btas::contract(1.0, iq_js, {1, 2, 3, 4}, C_uocc, {4, 5}, 0.0, iq_jb, {1, 2, 3, 5});
    btas::Tensor<double> ia_jb;
    btas::contract(1.0, iq_jb, {1, 2, 3, 4}, C_uocc, {2, 5}, 0.0, ia_jb, {1, 5, 3, 4});

    return ia_jb;
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

btas::Tensor<double> rei_ao_integrals_tensor(libint2::BasisSet& obs)
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
                                    //ao_ints.push_back( buf_1234[f1234] );
                                    rei_ao_ints(bf1, bf2, bf3, bf4) = buf_1234[f1234];
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
//TODO: creating equations for CCD

//    // iq_rs(i,q,r,s) = 1.0 * \sum_p pq_rs(p,q,r,s) * C_occ(p,i) + 0.0 * iq_rs(i,q,r,s)
//    btas::contract(1.0, pq_rs, {1, 2, 3, 4}, C_occ, {1, 5}, 0.0, iq_rs, {5, 2, 3, 4});
btas::Tensor<double> make_I_b_a(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> b_a; // (nuocc, nuocc)

    btas::Tensor<double> v_temp;
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    v_temp(i, a, j, b) = v_uu_oo(i, a, j, b) * -2 + v_uu_oo(i, b, j, a);
                }
            }
        }
    }

    btas::contract(1.0, v_temp, {1, 2, 3, 4}, cc_amplitudes, {1, 5, 3, 4}, 0.0, b_a, {2, 5}); //ambiguous final ordering? Less ambiguous than it used to be

    return b_a;
}

btas::Tensor<double> make_I_j_i(const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> j_i; // (nuocc, nuocc)

    btas::Tensor<double> v_temp;
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    v_temp(i, a, j, b) = 2 * v_uu_oo(i, a, j, b) - v_uu_oo(j, a, i, b);
                }
            }
        }
    }

    btas::contract(1.0, v_temp, {1, 2, 3, 4}, cc_amplitudes, {3, 5, 1, 2}, 0.0, j_i, {5, 4}); //ambiguous final ordering? Less ambiguous than it used to be
    return j_i;
}

btas::Tensor<double> make_I_kl_ij(const btas::Tensor<double>& v_oo_oo, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> kl_ij(nocc, nocc, nocc, nocc);
    //std::cout << "v dim: " << v.extent(0) << " " << v.extent(1) << " " << v.extent(2) << " " << v.extent(3) << std::endl;
    //std::cout << "cc_amps dim: " << cc_amplitudes.extent(0) << " " << cc_amplitudes.extent(1) << " " << cc_amplitudes.extent(2) << " " << cc_amplitudes.extent(3) << "\n";
    std::cout << "Initialized return vector I_ij_kl\n";
    btas::Tensor<double> vt;
    btas::contract(1.0, v_uu_oo, {1, 2, 3, 4}, cc_amplitudes, {1, 2, 5, 6}, 0.0, vt, {5, 6, 3, 4}); // as written ought to be {2, 5, 4, 6}, but this matches dim.
    std::cout << "First contraction I_ij_kl\n";
    //std::cout << "v dim: " << v.extent(0) << " " << v.extent(1) << " " << v.extent(2) << " " << v.extent(3) << std::endl;
    //std::cout << "vt dim: " << vt.extent(0) << " " << vt.extent(1) << " " << vt.extent(2) << " " << vt.extent(3) << "\n";
    kl_ij = v_oo_oo + vt; // sum dim not matching
    std::cout << "Finished sum\n";
    return kl_ij;
}

btas::Tensor<double> make_I_jb_ia(const btas::Tensor<double>& v_ou_ou, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n) { //Works
    //std::cout << "Inside make_I_ia_jb fn\n";
    int nuocc = n - nocc;
    btas::Tensor<double> ia_jb(nocc, nuocc, nocc, nuocc);
    //std::cout << "Initialized return vector\n";
    btas::Tensor<double> vt;
    btas::contract(0.5, v_uu_oo, {1, 2, 3, 4}, cc_amplitudes, {5, 4, 1, 6}, 0.0, vt, {5, 2, 3, 6});
    //std::cout << "First contraction\n";
    ia_jb = v_ou_ou - vt;
    return ia_jb;
}

btas::Tensor<double> make_I_bj_ia(const btas::Tensor<double>& v_uo_ou, const btas::Tensor<double>& v_uu_oo, const btas::Tensor<double>& cc_amplitudes, const int nocc, const int n) {
    int nuocc = n - nocc;
    btas::Tensor<double> ib_aj(nocc, nuocc, nuocc, nocc);
    std::cout << "Initialized return vector\n";
    //std::cout << "v dim: " << v.extent(0) << " " << v.extent(1) << " " << v.extent(2) << " " << v.extent(3) << std::endl;
    std::cout << "cc_amps dim: " << cc_amplitudes.extent(0) << " " << cc_amplitudes.extent(1) << " " << cc_amplitudes.extent(2) << " " << cc_amplitudes.extent(3) << "\n";
    btas::Tensor<double> vt_2ndterm;
    btas::contract(0.5, v_uu_oo, {1, 2, 3, 4}, cc_amplitudes, {5, 4, 1, 6}, 0.0, vt_2ndterm, {3, 2, 5, 6}); //ordering unsure, but more sure than it used to be // return {2, 3, 5, 6}
    std::cout << "First contraction\n";
    btas::Tensor<double> t_temp(nocc, nuocc, nocc, nuocc);
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    t_temp(i, a, j, b) = cc_amplitudes(i, a, j, b) - 0.5 * cc_amplitudes(i, b, j, a);
                    //std::cout << i << " " << a << " " << j << " " << b << "\n";
                }
            }
        }
    }
    std::cout << "Finished loop\n";

    btas::Tensor<double> vt_1stterm;
    btas::contract(1.0, v_uu_oo, {1, 2, 3, 4}, t_temp, {3, 4, 6, 5}, 0.0, vt_1stterm, {1, 2, 5, 6}); // {4, 5, 2, 6} -> {1, 3, 5, 6} // the t dims are wierd, should work by exchanging  3 & 4, 5 & 6
    std::cout << "Second contraction\n";
    //std::cout << "v dim: " << v.extent(0) << " " << v.extent(1) << " " << v.extent(2) << " " << v.extent(3) << std::endl;
    std::cout << "v1 dim: " << vt_1stterm.extent(0) << " " << vt_1stterm.extent(1) << " " << vt_1stterm.extent(2) << " " << vt_1stterm.extent(3) << std::endl;
    std::cout << "v2 dim: " << vt_2ndterm.extent(0) << " " << vt_2ndterm.extent(1) << " " << vt_2ndterm.extent(2) << " " << vt_2ndterm.extent(3) << std::endl;
    ib_aj = v_uo_ou + (vt_1stterm - vt_2ndterm); // could change contract for last term
    std::cout << "Vector subtraction\n";
    return ib_aj;
}

btas::Tensor<double> make_density_tensor(const Matrix& fock_matrix, const int nocc, const int n) { //called nao in main loop
    int nuocc = n - nocc;
    btas::Tensor<double> D(nocc, nuocc, nocc, nuocc);
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; a != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    D(i, a, j ,b) = fock_matrix(i, i) + fock_matrix(j, j) - fock_matrix(a, a) - fock_matrix(b, b);
                }
            }
        }
    }
    return D;
}

btas::Tensor<double>
        cc_amps_update(const btas::Tensor<double>& v_oo_uu, const btas::Tensor<double>& v_uu_uu, const btas::Tensor<double>& t,
                       const btas::Tensor<double>& I_a_b, const btas::Tensor<double>& I_i_j, const btas::Tensor<double>& I_ij_kl,
                       const btas::Tensor<double>& I_ia_jb, const btas::Tensor<double>& I_ia_bj, const int n, const int nocc) {

    int nuocc = n - nocc;
    btas::Tensor<double> Dt;

    //TODO: write each term in the sum (two permutations of indices plus v), then add and return
    btas::Tensor<double> t_ae_ij_I_b_e; // first term im permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_a_b, {5, 3}, 0.0, t_ae_ij_I_b_e, {1, 2, 5, 4});

    btas::Tensor<double> t_ab_im_I_m_j; // second term im permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_i_j, {4, 5}, 0.0, t_ab_im_I_m_j, {1, 2, 3, 5});

    btas::Tensor<double> v_ab_ef_t_ef_ij; // third term im permutative equation, note already scaled
    btas::contract(0.5, v, {1, 2, 3, 4}, t, {2, 5, 4, 6}, 0.0, v_ab_ef_t_ef_ij, {1, 5, 3, 6});

    btas::Tensor<double> t_ab_mn_I_mn_ij; // fourth term im permutative equation, note already scaled
    btas::contract(0.5, t, {1, 2, 3, 4}, I_ij_kl, {2, 5, 4, 6}, 0.0, t_ab_mn_I_mn_ij, {1, 5, 3, 6});

    btas::Tensor<double> t_ae_mj_I_mb_ie; // fifth term im permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_ia_jb, {2, 5, 6, 3}, 0.0, t_ae_mj_I_mb_ie, {1, 5, 6, 4});

    btas::Tensor<double> I_ma_ie_t_eb_mj; // sixth term im permutative equation
    btas::contract(1.0, I_ia_jb, {1, 2, 3, 4}, t, {4, 1, 5, 6}, 0.0, I_ma_ie_t_eb_mj, {3, 2, 5, 6});

    btas::Tensor<double> t_temp;
    for (int i = 0; i != nocc; ++i) {
        for (int j = 0; j != nocc; ++j) {
            for (int a = 0; j != nuocc; ++a) {
                for (int b = 0; b != nuocc; ++b) {
                    t_temp(a, i, b, j) = 2 * t(a, i, b, j) - t(a, j, b, i);
                }
            }
        }
    }

    btas::Tensor<double> t_ea_mi_I_mb_ej; // seventh term im permutative equation
    btas::contract(1.0, t_temp, {1, 2, 3, 4}, I_ia_bj, {2, 1, 5, 6}, 0.0, t_ea_mi_I_mb_ej, {3, 4, 5, 6});


    //TODO: Finish these permutated things
    btas::Tensor<double> t_ae_ij_I_b_e_perm; // first term im permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_a_b, {5, 3}, 0.0, t_ae_ij_I_b_e, {5, 4, 1, 2});

    btas::Tensor<double> t_ab_im_I_m_j_perm; // second term im permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_i_j, {4, 5}, 0.0, t_ab_im_I_m_j, {3, 5, 1, 2});

    btas::Tensor<double> v_ab_ef_t_ef_ij_perm; // third term im permutative equation, note already scaled
    btas::contract(0.5, v, {1, 2, 3, 4}, t, {2, 5, 4, 6}, 0.0, v_ab_ef_t_ef_ij, {3, 6, 1, 5});

    btas::Tensor<double> t_ab_mn_I_mn_ij_perm; // fourth term im permutative equation, note already scaled
    btas::contract(0.5, t, {1, 2, 3, 4}, I_ij_kl, {2, 5, 4, 6}, 0.0, t_ab_mn_I_mn_ij, {3, 6, 1, 5});

    btas::Tensor<double> t_ae_mj_I_mb_ie_perm; // fifth term im permutative equation
    btas::contract(1.0, t, {1, 2, 3, 4}, I_ia_jb, {2, 5, 6, 3}, 0.0, t_ae_mj_I_mb_ie, {6, 4, 1, 5});

    btas::Tensor<double> I_ma_ie_t_eb_mj_perm; // sixth term im permutative equation
    btas::contract(1.0, I_ia_jb, {1, 2, 3, 4}, t, {4, 1, 5, 6}, 0.0, I_ma_ie_t_eb_mj, {5, 6, 3, 2});

    btas::Tensor<double> t_ea_mi_I_mb_ej_perm; // seventh term im permutative equation
    btas::contract(1.0, t_temp, {1, 2, 3, 4}, I_ia_bj, {2, 1, 5, 6}, 0.0, t_ea_mi_I_mb_ej, {5, 6, 3, 4});

    Dt = v + t_ae_ij_I_b_e + t_ab_im_I_m_j + v_ab_ef_t_ef_ij + t_ab_mn_I_mn_ij + t_ae_mj_I_mb_ie + I_ma_ie_t_eb_mj + t_ea_mi_I_mb_ej +
            t_ae_ij_I_b_e_perm + t_ab_im_I_m_j_perm + v_ab_ef_t_ef_ij_perm + t_ab_mn_I_mn_ij_perm + t_ae_mj_I_mb_ie_perm +
            I_ma_ie_t_eb_mj_perm + t_ea_mi_I_mb_ej_perm;

    return Dt;
}

double calc_ccd_energy(btas::Tensor<double>& t, btas::Tensor<double> v, int n, int nocc) {
    double ccd_e = 0.0;
    auto nuocc = n - nocc;

    for (int j = 0; j != nocc; ++j) {
        for (int i = j + 1; i != nocc; ++i) {
            for (int b = 0; b != nuocc; ++b) {
                for (int a = b + 1; a != nuocc; ++a) {
                    ccd_e += v(i, j, a, b) * t(i, a, j, b);
                }
            }
        }
    }

    return ccd_e;
}