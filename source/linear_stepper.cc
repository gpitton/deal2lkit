//-----------------------------------------------------------
//
//    Copyright (C) 2015 - 2016 by the deal2lkit authors
//
//    This file is part of the deal2lkit library.
//
//    The deal2lkit library is free software; you can use it, redistribute
//    it, and/or modify it under the terms of the GNU Lesser General
//    Public License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//    The full text of the license can be found in the file LICENSE at
//    the top level of the deal2lkit distribution.
//
//-----------------------------------------------------------


#include <deal2lkit/linear_stepper.h>
#ifdef D2K_WITH_SUNDIALS

#include <deal.II/base/utilities.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/lac/block_vector.h>
#ifdef DEAL_II_WITH_TRILINOS
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#endif
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_parallel_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#endif
#include <deal.II/base/utilities.h>

#include <iostream>
#include <iomanip>

#include <math.h>

#ifdef DEAL_II_WITH_MPI
#include <nvector/nvector_parallel.h>
#endif

using namespace dealii;


D2K_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_MPI
template <typename VEC>
LinearStepper<VEC>::LinearStepper(std::string name,
                              MPI_Comm comm) :
  ParameterAcceptor(name),
  communicator(Utilities::MPI::duplicate_communicator(comm)),
  pcout(std::cout,
        Utilities::MPI::this_mpi_process(communicator)==0)
{
  set_functions_to_trigger_an_assert();
}

#else
template <typename VEC>
LinearStepper<VEC>::LinearStepper(std::string &name) :
  ParameterAcceptor(name),
  pcout(std::cout)
{
  set_functions_to_trigger_an_assert();
}
#endif

template <typename VEC>
LinearStepper<VEC>::~LinearStepper()
{
#ifdef DEAL_II_WITH_MPI
  MPI_Comm_free(&communicator);
#endif
}

template <typename VEC>
double LinearStepper<VEC>::get_alpha() const
{
  double alpha;
  if (initial_time == final_time || step_size == 0.0)
    alpha = 0.0;
  else
    alpha = 1./step_size;

  return alpha;
}

template <typename VEC>
void LinearStepper<VEC>::set_initial_time(const double &t)
{
  initial_time = t;
}

template <typename VEC>
void LinearStepper<VEC>::declare_parameters(ParameterHandler &prm)
{
  add_parameter(prm, &_step_size,
                "Step size", "1e-2", Patterns::Anything(),
                "Conditional statement can be used as well:\n"
                "(t<0.5?1e-2:1e-3)");

  add_parameter(prm, &stepper_type,
                "Stepper type", "Euler",
                Patterns::Anything());

  add_parameter(prm, &initial_time,
                "Initial time", "0.0",
                Patterns::Double());

  add_parameter(prm, &final_time,
                "Final time", "1.0",
                Patterns::Double());

  add_parameter(prm, &output_period,
                "Intervals between outputs", "1",
                Patterns::Integer());

  add_parameter(prm, &verbose,
                "Print useful informations", "false",
                Patterns::Bool());
}

template <typename VEC>
void LinearStepper<VEC>::compute_y_dot(const VEC &y, const VEC &prev, const double alpha, VEC &y_dot)
{
  y_dot = y;
  y_dot -= prev;
  y_dot *= alpha;
}

template <typename VEC>
unsigned int LinearStepper<VEC>::solve_dae(VEC &solution, VEC &solution_dot)
{
  unsigned int step_number = 0;

  double t = initial_time;
  double alpha;
  step_size = evaluate_step_size(t);
  alpha = get_alpha();
  // check if it is a stationary problem
  if (initial_time == final_time)
    alpha = 0.0;
  else if (stepper_type == "Euler")
    alpha = 1./step_size;


  bool restart=false;

  shared_ptr<VEC> residual_, zero_solution_dot_, previous_solution;

  // The overall cycle over time begins here.
  while (t<=final_time+1e-15)
    {
      pcout << "Solving for t = " << t
            << " (step size = "<< step_size<<")"
            << std::endl;

      restart = solver_should_restart(t,solution,solution_dot) || (t == initial_time);

      while (restart)
        {

          residual_ = create_new_vector();
          zero_solution_dot_ = create_new_vector(); // Always zero.

          restart = solver_should_restart(t,solution,solution_dot);
          setup_jacobian(t,
                         solution,
                         solution_dot,
                         alpha);

        }


      if (stepper_type=="Euler")
      {
        this->residual(t, solution, *zero_solution_dot_, *residual_);
        *residual_ *=-1;
        solve_jacobian_system(*residual_, solution_dot);
        solution += solution_dot;
        solution_dot /= step_size;
      }


      step_number += 1;

      if ((step_number % output_period) == 0)
        output_step(t, solution, solution_dot,  step_number);
      step_size = evaluate_step_size(t);
      t += step_size;

      if (initial_time == final_time)
        alpha = 0.0;
      else if (stepper_type=="Euler")
        alpha = 1./step_size;

      if (stepper_type != "Euler")
        *previous_solution = solution;

    } // End of the cycle over time.
  return 0;
}



template <typename VEC>
void LinearStepper<VEC>::
compute_previous_solution(const VEC &sol,
                          const VEC &sol_dot,
                          const double &alpha,
                          VEC &prev)
{
  if (alpha > 0.0)
    {
      prev = sol_dot;
      prev /= (-1.0*alpha);
      prev += sol;
    }
  else
    {
      prev = sol;
      (void)sol_dot;
      (void)alpha;
    }
}

template <typename VEC>
double LinearStepper<VEC>::evaluate_step_size(const double &t)
{
  std::string variables = "t";
  std::map<std::string,double> constants;
  // FunctionParser with 1 variables and 1 component:
  FunctionParser<1> fp(1);
  fp.initialize(variables,
                _step_size,
                constants);
  // Point at which we want to evaluate the function
  Point<1> time(t);
  // evaluate the expression at 'time':
  double result = fp.value(time);
  return result;
}



template<typename VEC>
void LinearStepper<VEC>::set_functions_to_trigger_an_assert()
{

  create_new_vector = []() ->shared_ptr<VEC>
  {
    shared_ptr<VEC> p;
    AssertThrow(false, ExcPureFunctionCalled("Please implement create_new_vector function."));
    return p;
  };

  residual = [](const double,
                const VEC &,
                const VEC &,
                VEC &) ->int
  {
    int ret=0;
    AssertThrow(false, ExcPureFunctionCalled("Please implement residual function."));
    return ret;
  };

  setup_jacobian = [](const double,
                      const VEC &,
                      const VEC &,
                      const double) ->int
  {
    int ret=0;
    AssertThrow(false, ExcPureFunctionCalled("Please implement setup_jacobian function."));
    return ret;
  };

  solve_jacobian_system = [](const VEC &,
                             VEC &) ->int
  {
    int ret=0;
    AssertThrow(false, ExcPureFunctionCalled("Please implement solve_jacobian_system function."));
    return ret;
  };

  output_step = [](const double,
                   const VEC &,
                   const VEC &,
                   const unsigned int)
  {
    AssertThrow(false, ExcPureFunctionCalled("Please implement output_step function."));
  };

  solver_should_restart = [](const double,
                             VEC &,
                             VEC &) ->bool
  {
    bool ret=false;
    AssertThrow(false, ExcPureFunctionCalled("Please implement solver_should_restart function."));
    return ret;
  };
}

D2K_NAMESPACE_CLOSE

template class deal2lkit::LinearStepper<BlockVector<double> >;

#ifdef DEAL_II_WITH_MPI

#ifdef DEAL_II_WITH_TRILINOS
template class deal2lkit::LinearStepper<TrilinosWrappers::MPI::Vector>;
template class deal2lkit::LinearStepper<TrilinosWrappers::MPI::BlockVector>;
#endif

#ifdef DEAL_II_WITH_PETSC
template class deal2lkit::LinearStepper<PETScWrappers::MPI::Vector>;
template class deal2lkit::LinearStepper<PETScWrappers::MPI::BlockVector>;
#endif

#endif

#endif
