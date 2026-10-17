#include <vector>
#include <random>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

#include <cstdio>
#include <cstdlib>
#include <cmath>

using real = double;

const real epsilon = 1e-6;

void print1dv(const std::vector<real>& aa)
{
  std::cerr << "[" << std::endl;
  for (real a : aa)
      std::cerr << std::fixed << std::setprecision(8) << a << std::endl;
  std::cerr << "]" << std::endl;
}

void print1dh(const std::vector<real>& aa)
{
  std::cerr << "[" << std::endl;
  for (real a : aa)
      std::cerr << std::fixed << std::setprecision(8) << a << ' ';
  std::cerr << std::endl << "]" << std::endl;
}

void print2d(const std::vector<real>& aa, int ma)
{
  int na = int(aa.size()) / ma;

  std::cerr << "[" << std::endl;
  for (int i = 0; i < na; i++) {
    for (int j = 0; j < ma; j++)
      std::cerr << std::fixed << std::setprecision(8) << aa[i * ma + j] << ' ';
    std::cerr << std::endl;
  }
  std::cerr << "]" << std::endl;
}

std::vector<real> add(const std::vector<real>& aa, int ma,
                      const std::vector<real>& bb, int mb)
{
  int na = int(aa.size()) / ma;
  int nb = int(bb.size()) / mb;
  if (na != nb || ma != mb) {
    std::fprintf(stderr,
      "matadd: size not equal (%d, %d) vs. (%d, %d)", na, ma, nb, mb);
    return {};
  }

  std::vector<real> cc(na * ma);
  for (int i = 0; i < na; i++)
    for (int j = 0; j < ma; j++)
      cc[i * ma + j] = aa[i * ma + j] + bb[i * ma + j];

  return cc;
}

std::vector<real> sub(const std::vector<real>& aa, int ma,
                      const std::vector<real>& bb, int mb)
{
  int na = int(aa.size()) / ma;
  int nb = int(bb.size()) / mb;
  if (na != nb || ma != mb) {
    std::fprintf(stderr,
      "matsub: size not equal (%d, %d) vs. (%d, %d)", na, ma, nb, mb);
    return {};
  }

  std::vector<real> cc(na * ma);
  for (int i = 0; i < na; i++)
    for (int j = 0; j < ma; j++)
      cc[i * ma + j] = aa[i * ma + j] - bb[i * ma + j];

  return cc;
}

std::vector<real> mul(real coeff,
                      const std::vector<real>& aa)
{
  std::vector<real> cc = aa;
  for (real& c : cc)
    c *= coeff;

  return cc;
}

std::vector<real> mul(const std::vector<real>& aa, int ma,
                      const std::vector<real>& bb, int mb)
{
  int na = int(aa.size()) / ma;
  int nb = int(bb.size()) / mb;
  if (ma != nb) {
    std::fprintf(stderr,
      "matmul: illegal (%d, %d) vs. (%d, %d)", na, ma, nb, mb);
    return {};
  }

  std::vector<real> cc(na * mb);
  for (int i = 0; i < na; i++)
    for (int j = 0; j < mb; j++)
      for (int k = 0; k < ma; k++)
        cc[i * mb + j] += aa[i * ma + k] * bb[k * mb + j];

  return cc;
}

std::vector<real> tra(const std::vector<real>& aa, int ma)
{
  int na = int(aa.size()) / ma;

  std::vector<real> cc(na * ma);
  for (int i = 0; i < na; i++)
    for (int j = 0; j < ma; j++)
      cc[j * ma + i] = aa[i * ma + j];

  return cc;
}

bool acceptable(const std::vector<real>& rr)
{
  for (real r : rr)
    if (std::abs(r) > epsilon)
      return false;
  return true;
}

std::vector<real> solve(const std::vector<real>& aa, int ma,
                        const std::vector<real>& bb)
{
  int na = int(aa.size()) / ma;
  if (na != ma) {
    std::fprintf(stderr,
      "solve: not square (%d, %d)\n", na, ma);
    return {};
  }

  int nb = int(bb.size());
  if (na != nb) {
    std::fprintf(stderr,
      "solve: illegal (%d, %d) vs. (%d, 1)\n", na, ma, nb);
    return {};
  }

  int n = na;

  std::vector<real> xx(n);

  /* https://en.wikipedia.org/wiki/Conjugate_gradient_method
   * >> Assume x_0 = [0] -> r_0 := b
   */
  std::vector<real> rr = bb;
  if (acceptable(rr))
    return xx;

  std::vector<real> pp = rr;
  int k = 0;
  while (true) {
    std::vector<real> alpha_top = mul(tra(rr, 1), n, rr, 1);
    std::vector<real> alpha_bot = mul(mul(tra(pp, 1), n, aa, n), n, pp, 1);
    real alpha = alpha_top[0] / alpha_bot[0];

    xx = add(xx, 1, mul(alpha, pp), 1);

    rr = sub(rr, 1, mul(mul(alpha, aa), n, pp, 1), 1);
    if (acceptable(rr))
      break;
   
    std::vector<real> beta_top = mul(tra(rr, 1), n, rr, 1);
    std::vector<real> beta_bot = alpha_top;
    real beta = beta_top[0] / beta_bot[0];

    pp = add(rr, 1, mul(beta, pp), 1);

    k++;
  }
  std::fprintf(stderr, ">> Solved in %d iterations\n", k);

  return xx;
}

bool check_positive_definite(const std::vector<real>& aa, int ma)
{
  int na = int(aa.size()) / ma;
  if (na != ma) {
    std::fprintf(stderr,
      "check: not square (%d, %d)\n", na, ma);
    return false;
  }

  for (int i = 0; i < na; i++) {
    for (int j = 0; j < ma; j++) {
      if (aa[i * ma + j] != aa[j * ma + i]) {
        std::fprintf(stderr,
          "check: not symmetric\n");
        return false;
      }
    }
  }

  /* https://www.gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/ */
  /* TODO: actually calculate the det of s x s top-left. */

  return true;
}

int main(int argc, char** argv)
{
  int n = std::atoi(argv[1]);

  std::vector<real> aa(n * n);
  std::vector<real> bb(n);

  /* https://www.geeksforgeeks.org/cpp/generate-random-double-numbers-in-cpp/ */
  /* https://stackoverflow.com/questions/60576072/c-uniform-real-distribution-keeps-outputting-the-same-values-each-time */
  real lower = -10;
  real upper =  10;
  std::random_device device;
  std::mt19937 rng(device());
  std::uniform_real_distribution<real> dist(lower, upper);
  do {
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++)
        aa[i * n + j] = aa[j * n + i] = dist(rng);
      bb[i] = dist(rng);
    }
  } while (!check_positive_definite(aa, n));

  std::fprintf(stderr, "[aa] = ");
  print2d(aa, n);

  std::fprintf(stderr, "[bb] = ");
  print1dv(bb);
  
  std::vector<real> xx = solve(aa, n, bb);
  std::fprintf(stderr, "[xx] = ");
  print1dv(xx);

  return 0;
}
