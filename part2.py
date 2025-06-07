import numpy as np
import matplotlib.pyplot as plt

class MM1QueueSimulation:
    def __init__(self, arrival_rate, service_rate, simulation_time):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.num_customers_served = 0
        self.total_system_time = 0
        self.total_queue_time = 0
        self.busy_time = 0
        self.area_system = 0
        self.area_queue = 0
        self.system_state_counts = {}
        self.prev_time = 0
        self.prev_system_size = 0
        self.prev_queue_size = 0
        self.arrival_times = []
        self.departure_times = []
        self.queue = []
        self.server_busy = False
        self.current_time = 0
        self.customer_id = 0
        self.customers_in_system = []

    def generate_interarrival_time(self):
        return np.random.exponential(1 / self.arrival_rate)

    def generate_service_time(self):
        return np.random.exponential(1 / self.service_rate)

    def update_statistics(self, new_time):
        time_elapsed = new_time - self.prev_time
        self.area_system += self.prev_system_size * time_elapsed
        self.area_queue += self.prev_queue_size * time_elapsed
        if self.prev_system_size not in self.system_state_counts:
            self.system_state_counts[self.prev_system_size] = 0
        self.system_state_counts[self.prev_system_size] += time_elapsed
        if self.server_busy:
            self.busy_time += time_elapsed
        self.prev_time = new_time

    def handle_arrival(self, next_departure_time):
        self.customer_id += 1
        arrival_time = self.current_time
        self.update_statistics(self.current_time)
        if not self.server_busy:
            self.server_busy = True
            service_time = self.generate_service_time()
            departure_time = self.current_time + service_time
            customer = {
                'id': self.customer_id,
                'arrival_time': arrival_time,
                'service_start_time': arrival_time,
                'departure_time': departure_time,
                'queue_time': 0,
                'system_time': service_time
            }
            self.customers_in_system.append(customer)
            return departure_time
        else:
            customer = {
                'id': self.customer_id,
                'arrival_time': arrival_time,
                'service_start_time': None,
                'departure_time': None,
                'queue_time': None,
                'system_time': None
            }
            self.queue.append(customer)
            self.customers_in_system.append(customer)
            return next_departure_time

    def handle_departure(self):
        self.update_statistics(self.current_time)
        departing_customer = None
        for i, customer in enumerate(self.customers_in_system):
            if (customer['departure_time'] is not None and
                    abs(customer['departure_time'] - self.current_time) < 1e-10):
                departing_customer = self.customers_in_system.pop(i)
                break
        if departing_customer:
            self.num_customers_served += 1
            self.total_system_time += departing_customer['system_time']
            self.total_queue_time += departing_customer['queue_time']
        if self.queue:
            next_customer = self.queue.pop(0)
            service_time = self.generate_service_time()
            departure_time = self.current_time + service_time
            next_customer['service_start_time'] = self.current_time
            next_customer['departure_time'] = departure_time
            next_customer['queue_time'] = self.current_time - next_customer['arrival_time']
            next_customer['system_time'] = departure_time - next_customer['arrival_time']
            return departure_time
        else:
            self.server_busy = False
            return float('inf')

    def get_current_system_size(self):
        return len(self.customers_in_system)

    def get_current_queue_size(self):
        return len(self.queue)

    def run_simulation(self):
        next_arrival = self.generate_interarrival_time()
        next_departure = float('inf')
        while self.current_time < self.simulation_time:
            self.prev_system_size = self.get_current_system_size()
            self.prev_queue_size = self.get_current_queue_size()
            if next_arrival < next_departure:
                self.current_time = next_arrival
                next_departure = self.handle_arrival(next_departure)
                next_arrival = self.current_time + self.generate_interarrival_time()
            else:
                self.current_time = next_departure
                next_departure = self.handle_departure()
        self.update_statistics(self.simulation_time)

    def get_performance_metrics(self):
        avg_system_size = self.area_system / self.simulation_time
        avg_queue_size = self.area_queue / self.simulation_time
        server_utilization = self.busy_time / self.simulation_time
        avg_system_time = self.total_system_time / self.num_customers_served if self.num_customers_served > 0 else 0
        avg_queue_time = self.total_queue_time / self.num_customers_served if self.num_customers_served > 0 else 0
        rho = self.arrival_rate / self.service_rate
        theoretical_avg_system_size = rho / (1 - rho) if rho < 1 else float('inf')
        theoretical_avg_queue_size = (rho ** 2) / (1 - rho) if rho < 1 else float('inf')
        theoretical_avg_system_time = 1 / (self.service_rate - self.arrival_rate) if rho < 1 else float('inf')
        theoretical_avg_queue_time = rho / (self.service_rate - self.arrival_rate) if rho < 1 else float('inf')
        return {
            'simulation_time': self.simulation_time,
            'customers_served': self.num_customers_served,
            'arrival_rate': self.arrival_rate,
            'service_rate': self.service_rate,
            'utilization': rho,
            'sim_avg_system_size': avg_system_size,
            'sim_avg_queue_size': avg_queue_size,
            'sim_server_utilization': server_utilization,
            'sim_avg_system_time': avg_system_time,
            'sim_avg_queue_time': avg_queue_time,
            'theo_avg_system_size': theoretical_avg_system_size,
            'theo_avg_queue_size': theoretical_avg_queue_size,
            'theo_server_utilization': rho,
            'theo_avg_system_time': theoretical_avg_system_time,
            'theo_avg_queue_time': theoretical_avg_queue_time,
            'system_state_counts': self.system_state_counts
        }

    def plot_results(self, metrics):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        states = list(metrics['system_state_counts'].keys())
        probabilities = [metrics['system_state_counts'][state] / self.simulation_time for state in states]
        rho = metrics['utilization']
        if rho < 1:
            theo_probs = [(1 - rho) * (rho ** state) for state in states]
        else:
            theo_probs = [0] * len(states)
        ax1.bar([s - 0.2 for s in states], probabilities, 0.4, label='Simulated', alpha=0.7)
        ax1.bar([s + 0.2 for s in states], theo_probs, 0.4, label='Theoretical', alpha=0.7)
        ax1.set_xlabel('Number of Customers in System')
        ax1.set_ylabel('Probability')
        ax1.set_title('System Size Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        metrics_names = ['Avg System Size', 'Avg Queue Size', 'Server Utilization']
        sim_values = [metrics['sim_avg_system_size'], metrics['sim_avg_queue_size'], metrics['sim_server_utilization']]
        theo_values = [metrics['theo_avg_system_size'], metrics['theo_avg_queue_size'], metrics['theo_server_utilization']]
        x = np.arange(len(metrics_names))
        width = 0.35
        ax2.bar(x - width / 2, sim_values, width, label='Simulated', alpha=0.7)
        ax2.bar(x + width / 2, theo_values, width, label='Theoretical', alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Values')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        time_metrics = ['Avg System Time', 'Avg Queue Time']
        sim_time_values = [metrics['sim_avg_system_time'], metrics['sim_avg_queue_time']]
        theo_time_values = [metrics['theo_avg_system_time'], metrics['theo_avg_queue_time']]
        x_time = np.arange(len(time_metrics))
        ax3.bar(x_time - width / 2, sim_time_values, width, label='Simulated', alpha=0.7)
        ax3.bar(x_time + width / 2, theo_time_values, width, label='Theoretical', alpha=0.7)
        ax3.set_xlabel('Time Metrics')
        ax3.set_ylabel('Time (minutes)')
        ax3.set_title('Time Metrics Comparison')
        ax3.set_xticks(x_time)
        ax3.set_xticklabels(time_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax4.axis('tight')
        ax4.axis('off')
        table_data = [
            ['Metric', 'Simulated', 'Theoretical', 'Error %'],
            ['Avg System Size', f'{metrics["sim_avg_system_size"]:.3f}',
             f'{metrics["theo_avg_system_size"]:.3f}',
             f'{abs(metrics["sim_avg_system_size"] - metrics["theo_avg_system_size"]) / metrics["theo_avg_system_size"] * 100:.2f}%' if
             metrics["theo_avg_system_size"] != float('inf') else 'N/A'],
            ['Avg Queue Size', f'{metrics["sim_avg_queue_size"]:.3f}',
             f'{metrics["theo_avg_queue_size"]:.3f}',
             f'{abs(metrics["sim_avg_queue_size"] - metrics["theo_avg_queue_size"]) / metrics["theo_avg_queue_size"] * 100:.2f}%' if
             metrics["theo_avg_queue_size"] != float('inf') else 'N/A'],
            ['Server Utilization', f'{metrics["sim_server_utilization"]:.3f}',
             f'{metrics["theo_server_utilization"]:.3f}',
             f'{abs(metrics["sim_server_utilization"] - metrics["theo_server_utilization"]) / metrics["theo_server_utilization"] * 100:.2f}%'],
            ['Avg System Time', f'{metrics["sim_avg_system_time"]:.3f}',
             f'{metrics["theo_avg_system_time"]:.3f}',
             f'{abs(metrics["sim_avg_system_time"] - metrics["theo_avg_system_time"]) / metrics["theo_avg_system_time"] * 100:.2f}%' if
             metrics["theo_avg_system_time"] != float('inf') else 'N/A']
        ]
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary', pad=20)
        plt.tight_layout()
        plt.show()

def run_example(arrival_rate=4, service_rate=12, simulation_time=10000):
    np.random.seed(42)
    arrival_rate = arrival_rate/60
    service_rate = service_rate/60
    simulation_time = simulation_time
    print(f"M/M/1 Queue Simulation")
    print(f"Arrival rate (λ): {arrival_rate} customers/min")
    print(f"Service rate (μ): {service_rate} customers/min")
    print(f"Utilization (ρ): {arrival_rate / service_rate:.3f}")
    print(f"Simulation time: {simulation_time} minutes")
    print("-" * 50)
    sim = MM1QueueSimulation(arrival_rate, service_rate, simulation_time)
    sim.run_simulation()
    metrics = sim.get_performance_metrics()
    print(f"Customers served: {metrics['customers_served']}")
    print(f"\nSimulated vs Theoretical Results:")
    print(f"Average system size: {metrics['sim_avg_system_size']:.3f} vs {metrics['theo_avg_system_size']:.3f}")
    print(f"Average queue size: {metrics['sim_avg_queue_size']:.3f} vs {metrics['theo_avg_queue_size']:.3f}")
    print(f"Server utilization: {metrics['sim_server_utilization']:.3f} vs {metrics['theo_server_utilization']:.3f}")
    print(f"Average system time: {metrics['sim_avg_system_time']:.3f} vs {metrics['theo_avg_system_time']:.3f}")
    print(f"Average queue time: {metrics['sim_avg_queue_time']:.3f} vs {metrics['theo_avg_queue_time']:.3f}")
    sim.plot_results(metrics)
    return sim, metrics

if __name__ == "__main__":
    simulation, results = run_example(6,12)