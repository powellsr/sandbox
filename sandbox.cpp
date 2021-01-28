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

double mp2_energy(const Matrix&C, const btas::Tensor<double>& ao_ints, int nocc, int n, const Eigen::VectorXd& evals);
btas::Tensor<double> transform_pqrs_to_iajb(const Matrix& C, const btas::Tensor<double>& pq_rs, int nocc);
btas::Tensor<double> rei_ao_integrals_tensor(libint2::BasisSet& obs);

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
    do {
      const auto tstart = std::chrono::high_resolution_clock::now();
      ++iter;

      // Save a copy of the energy and the density
      auto ehf_last = ehf;
      auto D_last = D;

      // build a new Fock matrix
      auto F = H;
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
    /*for (int i = 0; i< ao_ints.size(); ++i) {
        if (i % 7 == 0)
            std::cout << "\n";
        total_vector_sum += ao_ints[i];
        printf("%10.6f\t", ao_ints[i]);
    }
    std::cout << std::endl;
    */
    btas::Tensor<double> ao_ints_ten = rei_ao_integrals_tensor(obs);
    auto mp2_e_ten = mp2_energy(C, ao_ints_ten, ndocc, nao, eps);

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

    /* // Contractions via loop
    btas::Tensor<double> iq_rs_copy(nocc, n, n, n);
    iq_rs_copy.fill(0);
    double temp;
    for (int i = 0; i != nocc; ++i) {
        for (int q = 0; q != n; ++q) {
            for (int r = 0; r != n; ++r) {
                for (int s = 0; s != n; ++s) {
                    temp = 0.0;
                    for (int p = 0; p != n; ++p) {
                        temp += 1.0 * pq_rs(p,q,r,s) * C_occ(p,i); // + 0.0 * iq_rs(i,q,r,s)
                    }
                    iq_rs_copy(i,q,r,s) = temp;
                }
            }
        }
    }
    btas::Tensor<double> iq_js_copy(nocc, n, nocc, n);
    iq_js_copy.fill(0);
    for (int i = 0; i != nocc; ++i) {
        for (int q = 0; q != n; ++q) {
            for (int j = 0; j != nocc; ++j) {
                for (int s = 0; s != n; ++s) {
                    temp = 0.0;
                    for (int r = 0; r != n; ++r) {
                        temp += 1.0 * iq_rs_copy(i,q,r,s) * C_occ(r,j);
                    }
                    iq_js_copy(i,q,j,s) = temp;
                }
            }
        }
    }
    btas::Tensor<double> iq_jb_copy(nocc, n, nocc, nuocc);
    iq_jb_copy.fill(0);
    for (int i = 0; i != nocc; ++i) {
        for (int q = 0; q != n; ++q) {
            for (int j = 0; j != nocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    temp = 0.0;
                    for (int s = 0; s != n; ++s) {
                        temp += 1.0 * iq_js_copy(i,q,j,s) * C_uocc(s,b);
                    }
                    iq_jb_copy(i,q,j,b) = temp;
                }
            }
        }
    }
    btas::Tensor<double> ia_jb_copy(nocc, nuocc, nocc, nuocc);
    ia_jb_copy.fill(0);
    for (int i = 0; i != nocc; ++i) {
        for (int a = 0; a != nuocc; ++a) {
            for (int j = 0; j != nocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    temp = 0.0;
                    for (int q = 0; q != n; ++q) {
                        temp += 1.0 * iq_jb_copy(i,q,j,b) * C_uocc(q,a);
                    }
                    iq_jb_copy(i,a,j,b) = temp;
                }
            }
        }
    }*/

    btas::Tensor<double> iq_rs;  // (iq|rs)
    // iq_rs(i,q,r,s) = 1.0 * \sum_p pq_rs(p,q,r,s) * C_occ(p,i) + 0.0 * iq_rs(i,q,r,s)
    btas::contract(1.0, pq_rs, {1, 2, 3, 4}, C_occ, {1, 5}, 0.0, iq_rs, {5, 2, 3, 4});
    btas::Tensor<double> iq_js;
    btas::contract(1.0, iq_rs, {1, 2, 3, 4}, C_occ, {3, 5}, 0.0, iq_js, {1, 2, 5, 4});
    btas::Tensor<double> iq_jb;
    btas::contract(1.0, iq_js, {1, 2, 3, 4}, C_uocc, {4, 5}, 0.0, iq_jb, {1, 2, 3, 5});
    btas::Tensor<double> ia_jb;
    btas::contract(1.0, iq_jb, {1, 2, 3, 4}, C_uocc, {2, 5}, 0.0, ia_jb, {1, 5, 3, 4});

/*  // Comparison of loop and btas::contract contractions
    for (int i = 0; i != nocc; ++i) {
        for (int q = 0; q != n; ++q) {
            for (int r = 0; r != n; ++r) {
                for (int s = 0; s != n; ++s) {
                    if (std::abs(iq_rs(i,q,r,s) - iq_rs_copy(i,q,r,s)) > (1e-15))
                        std::cout << iq_rs(i,q,r,s) - iq_rs_copy(i,q,r,s) << std::endl;
                }
            }
        }
    }
    std::cout << "evaluate second contraction: \n";
    for (int i = 0; i != nocc; ++i) {
        for (int q = 0; q != n; ++q) {
            for (int j = 0; j != nocc; ++j) {
                for (int s = 0; s != n; ++s) {
                    if (std::abs(iq_js(i,q,j,s) - iq_js_copy(i,q,j,s)) > (1e-15))
                        std::cout << iq_js(i,q,j,s) - iq_js_copy(i,q,j,s) << std::endl;
                }
            }
        }
    }
    std::cout << "evaluate third contraction: \n";
    for (int i = 0; i != nocc; ++i) {
        for (int q = 0; q != n; ++q) {
            for (int j = 0; j != nocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    if (std::abs(iq_jb(i,q,j,b) - iq_jb_copy(i,q,j,b)) > (1e-15))
                        std::cout << iq_jb(i,q,j,b) - iq_jb_copy(i,q,j,b) << std::endl;
                }
            }
        }
    }
    std::cout << "evaluate fourth contraction: \n";
    for (int i = 0; i != nocc; ++i) {
        for (int a = 0; a != nuocc; ++a) {
            for (int j = 0; j != nocc; ++j) {
                for (int b = 0; b != nuocc; ++b) {
                    if (std::abs(ia_jb(i,a,j,b) - ia_jb_copy(i,a,j,b)) > (1e-15))
                        std::cout << ia_jb(i,a,j,b) - ia_jb_copy(i,a,j,b) << std::endl;
                }
            }
        }
    }
    */

    return ia_jb;
}

double mp2_energy(const Matrix& C, const btas::Tensor<double>& pq_rs, int nocc, int n, const Eigen::VectorXd& evals) {
  double mp2e = 0.0;

  auto ia_jb = transform_pqrs_to_iajb(C, pq_rs, nocc);

  const auto nuocc = n - nocc;
  for (int i = 0; i < nocc; ++i) {
    for (int j = 0; j < nocc; ++j) {
      for (int a = 0; a < nuocc; ++a) {
        for (int b = 0; b < nuocc; ++b) {
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