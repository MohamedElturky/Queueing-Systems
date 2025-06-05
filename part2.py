from collections import defaultdict
import numpy as np

class Simulation:
    def __init__(self, arrival_rate, service_rate):
        """
        :param arrival_rate: λ (customers/hour)
        :param service_rate: µ (customers/hour)
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate

    def simulate(self, time=100000):
        np.random.seed(0)
        λ = self.arrival_rate / 60  # per minute
        µ = self.service_rate / 60  # per minute

        current_time = 0
        last_departure = 0

        arrival_times = []
        service_start_times = []
        departure_times = []
        queue_lengths = []
        system_lengths = []
        state_counter = defaultdict(int)

        busy_time = 0
        idle_time = 0

        while current_time < time:
            inter_arrival = np.random.exponential(1 / λ)
            service_time = np.random.exponential(1 / µ)

            arrival_time = current_time + inter_arrival
            start_time = max(arrival_time, last_departure)
            departure_time = start_time + service_time

            # Metrics tracking
            wait_time = start_time - arrival_time
            time_in_system = departure_time - arrival_time

            arrival_times.append(arrival_time)
            service_start_times.append(start_time)
            departure_times.append(departure_time)

            busy_time += service_time
            idle_time += max(0, arrival_time - last_departure)
            last_departure = departure_time
            current_time = arrival_time

            # Track number of people in system over time
            system_lengths.append(time_in_system)
            queue_lengths.append(wait_time)
            n_in_system = sum([1 for t in departure_times if t > arrival_time])
            state_counter[n_in_system] += 1

        total_customers = len(arrival_times)
        total_time = departure_times[-1]

        # Metrics
        L = np.mean([n for n in state_counter.keys() for _ in range(state_counter[n])])
        Lq = np.mean(queue_lengths) * λ
        Ws = np.mean(system_lengths)
        Wq = np.mean(queue_lengths)
        ρ = busy_time / total_time

        # Empirical state probabilities
        total_state_obs = sum(state_counter.values())
        state_probs = {n: count / total_state_obs for n, count in sorted(state_counter.items())}

        # Output
        print(f"--- Simulation Results for λ = {self.arrival_rate}, µ = {self.service_rate} ---")
        print(f"Utilization (ρ): {ρ:.4f}")
        print(f"Average number in system (L): {L:.4f}")
        print(f"Average number in queue (Lq): {Lq:.4f}")
        print(f"Average time in system (Ws): {Ws * 60:.2f} minutes")
        print(f"Average time in queue (Wq): {Wq * 60:.2f} minutes")
        print(f"State probabilities (Pn):")
        for n, p in state_probs.items():
            print(f"  P({n}) = {p:.4f}")


sim = Simulation(arrival_rate=4, service_rate=12)
sim.simulate()
