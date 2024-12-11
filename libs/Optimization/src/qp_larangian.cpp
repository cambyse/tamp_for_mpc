#include <Optimization/qp_lagrangian.h>

namespace
{
void setToZeroWhereNegative(arr& g, arr& Jg)
{
  for(uint i = 0; i < g.d0; ++i)
  {
    if(g(i) < 0.0)
    {
      g(i) = 0.0;

      for(auto j = 0; j < Jg.d1; ++j)
      {
        Jg(i,j) = 0;
      }
    }
  }
}
}

QP_Problem::QP_Problem(const arr& P, const arr& q, const arr& K, const arr& u)
  : P(P)
  , q(q)
  , K(K)
  , u(u)
{
  transpose(qT, q);
}

double QP_Problem::operator()(arr& dL, arr& HL, const arr& x) const
{
  arr xT;
  transpose(xT, x);

  auto v = 0.5 * xT * P * x + qT * x;

  if(!!dL)
  {
    dL = P * x + q;
  }

  if(!!HL)
  {
    HL = P;
  }

  return v(0);
}

QP_Lagrangian::QP_Lagrangian(const QP_Problem& P, OptOptions opt, arr& lambdaInit)
  : qp(P)
  , mu(opt.muInit)
{
  lambda = zeros(qp.K.d0);

  ScalarFunction::operator=([this](arr& dL, arr& HL, const arr& x) -> double {
    return this->lagrangian(dL, HL, x);
  });
}

double QP_Lagrangian::lagrangian(arr& dL, arr& HL, const arr& _x) ///< CORE METHOD: the unconstrained scalar function F
{
  if(_x!=x)
  {
    x = _x;
  }

  double L = c = qp(dL, HL, x);

  // handle inequality constraints
  if(qp.K.d0)
  {
    g = qp.K * x - qp.u; // Jg = K

    ///  lagrange term
    L += scalarProduct(lambda, g);
    // jacobian
    if(!!dL)
    {
      const auto& Jg = qp.K;
      dL += comp_At_x(lambda, Jg);
    }

    /// square penalty
    auto Jg = qp.K;
    setToZeroWhereNegative(g, Jg);

    L += mu * scalarProduct(g, g);

    // jacobian
    if(!!dL)
    {
      dL += 2.0 * mu * comp_At_x(g, Jg);
    }

    // hessian
    if(!!HL)
    {
      HL += 2.0 * mu * comp_At_A(Jg); // computation of the gram product of the jacobian!
      // idea: copute the hessian of each constraint separately only once, and only sum the precomputed hessian!
    }
  }

//  std::cout << "Jg:" << qp.K.d0 << " " << qp.K.d1 << std::endl;
//  std::cout << "Jacobian:" << dL.d0 << " " << dL.d1 << std::endl;
//  std::cout << "Hessian:" << HL.d0 << " " << HL.d1 << std::endl;

  return L;
}

double QP_Lagrangian::get_costs()
{
  return c;
}

double QP_Lagrangian::get_sumOfGviolations()
{
  return sum(g); // asuming it is called after the lagrangian, the negative parts are already set to zero :)
}

double QP_Lagrangian::get_sumOfHviolations()
{
  return 0.0;
}

uint QP_Lagrangian::get_dimOfType(const ObjectiveType& tt)
{
  if(tt == OT_ineq)
    return lambda.d0;
  if(tt == OT_eq)
    return 0;
  if(tt == OT_sos)
    return 1;
  return 0;
}

void QP_Lagrangian::aulaUpdate(bool anyTimeVariant, double _, double muInc, double *L_x, arr &dL_x, arr &HL_x)
{
  for(auto i = 0; i < lambda.d0; ++i)
  {
    lambda(i) = std::max(lambda(i) + 2.0 * mu * g(i), 0.0);
    // see "A tutorial on Newton methods for constrained trajectory optimization and relations to SLAM,
    // Gaussian Process smoothing, optimal control, and probabilistic inference" p.7

    // if g(i) negative, the value of lambda will decrease (less need to push away ro the constraint!).
    // lambda(i) can't become negative though as it would mean pushing towards the constraints
  }

  if(muInc>1. && mu<1e5) mu *= muInc;

  //-- recompute the Lagrangian with the new parameters (its current value, gradient & hessian)
  if(L_x || !!dL_x || !!HL_x)
  {
    double L = lagrangian(dL_x, HL_x, x); //re-evaluate gradients and hessian (using buffered info)
    if(L_x) *L_x = L;
  }
}
