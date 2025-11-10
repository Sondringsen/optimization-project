from niapy.algorithms.basic import ComprehensiveLearningParticleSwarmOptimizer

class CLPSO_wrapper(ComprehensiveLearningParticleSwarmOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Code copied from niapy.algorithms.basic.ComprehensiveLearningParticleSwarmOptimizer.run_iteration
    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current populations.
            fpop (numpy.ndarray): Current population fitness/function values.
            xb (numpy.ndarray): Current best particle.
            fxb (float): Current best particle fitness/function value.
            params (dict): Additional function keyword arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list, dict]:
                1. New population.
                2. New population fitness/function values.
                3. New global best position.
                4. New global best positions function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments:
                    * personal_best: Particles best population.
                    * personal_best_fitness: Particles best positions function/fitness value.
                    * min_velocity: Minimal velocity.
                    * max_velocity: Maximal velocity.
                    * V: Initial velocity of particle.
                    * flag: Refresh gap counter.
                    * pc: Learning rate.

        See Also:
            * :class:`niapy.algorithms.basic.ParticleSwarmAlgorithm.run_iteration`

        """
        personal_best = params.pop('personal_best')
        personal_best_fitness = params.pop('personal_best_fitness')
        min_velocity = params.pop('min_velocity')
        max_velocity = params.pop('max_velocity')
        v = params.pop('v')
        flag = params.pop('flag')
        pc = params.pop('pc')

        w = self.w0 * (self.w0 - self.w1) * (task.iters + 1) / task.max_iters
        for i in range(len(pop)):
            if flag[i] >= self.m:
                v[i] = self.update_velocity(v[i], pop[i], personal_best[i], xb, 1, min_velocity, max_velocity, task)
                pop[i] = task.repair(pop[i] + v[i], rng=self.rng)
                fpop[i] = task.eval(pop[i])
                if fpop[i] < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
                    if fpop[i] < fxb:
                        xb, fxb = pop[i].copy(), fpop[i]
                flag[i] = 0
            pbest = self.generate_personal_best_cl(i, pc[i], personal_best, personal_best_fitness)
            v[i] = self.update_velocity_cl(v[i], pop[i], pbest, w, min_velocity, max_velocity, task)
            pop[i] = pop[i] + v[i]
            if task.is_feasible(pop[i]):
                fpop[i] = task.eval(pop[i])
                if fpop[i] < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
                    if fpop[i] < fxb:
                        xb, fxb = pop[i].copy(), fpop[i]
        return pop, fpop, xb, fxb, {'personal_best': personal_best, 'personal_best_fitness': personal_best_fitness,
                                    'min_velocity': min_velocity,
                                    'max_velocity': max_velocity, 'v': v, 'flag': flag, 'pc': pc}