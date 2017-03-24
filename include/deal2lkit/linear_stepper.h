//-----------------------------------------------------------
//
//    Copyright (C) 2015-2016 by the deal2lkit authors
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

#ifndef _d2k_linear_stepper_h
#define _d2k_linear_stepper_h

#include <deal2lkit/config.h>
#include <deal.II/base/config.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/conditional_ostream.h>

#ifdef D2K_WITH_SUNDIALS
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/kinsol_interface.h>

#ifdef DEAL_II_WITH_MPI
#include "mpi.h"
#endif

D2K_NAMESPACE_OPEN

/**
 * LinearStepper solves non-linear time dependent problems with
 * user-defined size of the time step and using Newthon's
 * method for the solution of the non-linear problem.
 * It allows to use the Kinsol solver of \sundials.
 *
 * The user has to provide the following std::functions:
 *  - create_new_vector;
 *  - residual;
 *  - setup_jacobian;
 *  - solve_jacobian_system;
 *  - output_step;
 *  - solver_should_restart;
 *
 */
template<typename VEC=Vector<double> >
class LinearStepper : public ParameterAcceptor
{
public:

#ifdef DEAL_II_WITH_MPI
  /** Constructor for the LinearStepper class.
    * Takes a @p name for the section in parameter file
    * and a mpi communicator.
    */
  LinearStepper(std::string name="",
              MPI_Comm comm = MPI_COMM_WORLD);
#else
  /** Constructor for the LinearStepper class.
    * Takes a @p name for the section in parameter file.
    */
  LinearStepper(std::string &name="");

#endif

  ~LinearStepper();

  /** Declare parameters for this class to function properly. */
  virtual void declare_parameters(ParameterHandler &prm);

  /** Evolve. This function returns the final number of steps. */
  unsigned int solve_dae(VEC &solution, VEC &solution_dot);

  /**
     * if initial time is different from final time (i.e.,
     * we are solving a time-dep problem and not a stationay
     * one, return the inverse of dt. If the problem is
     * stationary, returns 0.
     * @return
     */
  double get_alpha() const;

  /**
       * Set initial time equal to @p t disregarding what
       * is written in the parameter file.
       */
  void set_initial_time(const double &t);

private:

#ifdef DEAL_II_WITH_MPI
  MPI_Comm communicator;
#endif

  void compute_y_dot(const VEC &y, const VEC &prev, const double alpha, VEC &y_dot);

  /** Step size. */
  double step_size;

  /**
     * user defined step_size
     */
  std::string _step_size;

  /**
    * Time stepper to use.
    */
  std::string stepper_type;

  /**
    * @brief evaluate step size at time @p t according to the
    * expression stored in _step_size
    */
  double evaluate_step_size(const double &t);

  /** Initial time for the ode.*/
  double initial_time;

  /** Final time. */
  double final_time;

  /** Seconds between each output. */
  unsigned int output_period;

  /** Output stream */
  ConditionalOStream pcout;

  /** print useful informations */
  bool verbose;

  /** advance a single time step */
  void advance_step(shared_ptr<VEC> &res,
                    shared_ptr<VEC> &rhs,
                    const shared_ptr<VEC> &previous_solution,
                    VEC &solution);

  /**
   * compute previous solution from given
   * @param sol
   * @param sol_dot
   * @param alpha
   */
  void compute_previous_solution(const VEC &sol,
                                 const VEC &sol_dot,
                                 const double &alpha,
                                 VEC &prev);

public:

  /**
   * Return a shared_ptr<VEC>. A shared_ptr is needed in order
   * to keep the pointed vector alive, without the need to use a
   * static variable.
   */
  std::function<shared_ptr<VEC>()> create_new_vector;

  /**
   * Compute residual.
   */
  std::function<int(const double t,
                    const VEC &y,
                    const VEC &y_dot,
                    VEC &res)> residual;

  /**
   * Compute Jacobian.
   */
  std::function<int(const double t,
                    const VEC &y,
                    const VEC &y_dot,
                    const double alpha)> setup_jacobian;

  /**
   * Solve linear system.
   */
  std::function<int(const VEC &rhs, VEC &dst)> solve_jacobian_system;

  /**
   * Store solutions to file.
   */
  std::function<void (const double t,
                      const VEC &sol,
                      const VEC &sol_dot,
                      const unsigned int step_number)> output_step;

  /**
   * Evaluate wether the mesh should be refined or not. If so, it
   * refines and interpolate the solutions from the old to the new
   * mesh.
   */
  std::function<bool (const double t,
                      VEC &sol,
                      VEC &sol_dot)> solver_should_restart;

private:

  /**
   * Set the std::functions above to trigger an assert if they are not implemented.
   */
  void set_functions_to_trigger_an_assert();

};

D2K_NAMESPACE_CLOSE

#endif


#endif
