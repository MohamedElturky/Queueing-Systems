class TheoreticalCalculation:
    def __init__(self, arrival_rate=None, service_rate=None, utilization_factor=None):
        """
        at least set 2 attributes

        :param arrival_rate: λ (customers/hour)
        :param service_rate: µ (customers/hour)
        :param utilization_factor:
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        if [arrival_rate, service_rate, utilization_factor].count(None) > 1:
            raise ValueError("The arrival rate and departure rates must not be None")
        elif utilization_factor is None:
            self.utilization_factor = arrival_rate / service_rate
        elif arrival_rate is None:
            self.arrival_rate = utilization_factor * service_rate
        elif service_rate is None:
            self.service_rate = arrival_rate / utilization_factor

        self.avg_number_customers_system = self.utilization_factor/ (1-self.utilization_factor)
        self.avg_number_customers_queue = self.avg_number_customers_system * self.utilization_factor
        self.avg_time_customers_system = 1/(self.service_rate *(1-self.utilization_factor))
        self.avg_time_customers_queue = self.avg_time_customers_system * self.utilization_factor
        self.state_probability=[(1-self.utilization_factor)*pow(self.utilization_factor,i) for i in range(10)]

x = TheoreticalCalculation(4,12)
print(x.utilization_factor)

print(x.avg_number_customers_system)
print(x.avg_number_customers_queue)

print(x.avg_time_customers_system*60)
print(x.avg_time_customers_queue*60)

print(x.state_probability)






