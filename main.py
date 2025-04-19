from Mass_v3 import model


parameters = {
    'm' : 500e-3, # kg
    'k' : 10, # N/m
    'kh' : 0.2, # N/m
    'gamma' : 1, # N/ m/s
    'mu' : 0.35,
}

o1 = model(**parameters)



steps = ([1]*10, [10]*10)

o1.integrate(steps)