import numpy as np

def get_properties_gas(species_list):
    MW = []
    mu = []
    Dm = []
    Cp_molar = []
    Cp_mass = []
    k = []
    H = []

    for gas in species_list:
        if gas not in gases:
            raise ValueError(f"Gas '{gas}' no encontrado en la base de datos.")
        
        props = gases[gas]
        MW.append(props["MW"])
        mu.append(props["mu"])
        Dm.append(props["Dm"])
        Cp_molar.append(props["Cp_molar"])
        Cp_mass.append(props["Cp_mass"])
        k.append(props["k"])
        H.append(props["H"])
    
    return (np.array(MW),
            np.array(mu),
            np.array(Dm),
            np.array(Cp_molar),
            np.array(Cp_mass),
            np.array(k),
            np.array(H))
    

gases = {
    
    "CO2": {
        "MW": 0.04400995,            # kg/mol
        "mu": 1.37e-5,               # Pa·s
        "Dm": 1.3e-5,                # m²/s
        "Cp_mass": 840.37,          # J/kg·K
        "Cp_molar": 36.98464168,    # J/mol·K
        "k": 0.0145,                 # W/m·K
        "H": -3.935e8               # J/kmol
    },
    
    "CO": {
        "MW": 0.02801055,
        "mu": 1.75e-5,
        "Dm": 1.3e-5,
        "Cp_mass": 1043.0,
        "Cp_molar": 29.21500365,
        "k": 0.025,
        "H": -1.105e8
    },
    
    "N2": {
        "MW": 0.0280134,
        "mu": 1.663e-5,
        "Dm": 1.3e-5,
        "Cp_mass": 1040.67,
        "Cp_molar": 29.15270498,
        "k": 0.0242,
        "H": 0.0
    },
    
    "O2": {
        "MW": 0.031998,
        "mu": 2.01e-5,
        "Dm": 1.3e-5,
        "Cp_mass": 918.0,
        "Cp_molar": 29.37416423,
        "k": 0.0264,
        "H": 0.0
    },
    
    "H2": {
        "MW": 0.00201594,
        "mu": 8.411e-6,
        "Dm": 1.3e-5,
        "Cp_mass": 14283.0,
        "Cp_molar": 28.79367102,
        "k": 0.1672,
        "H": 0.0
    },
    
    "H2S": {
        "MW": 0.03407994,
        "mu": 1.20e-5,
        "Dm": 1.3e-5,
        "Cp_mass": 1170.0,
        "Cp_molar": 39.8735298,
        "k": 0.0134,
        "H": -2.051e7
    },
    
    "CH4": {
        "MW": 0.01604303,
        "mu": 1.087e-5,
        "Dm": 1.3e-5,
        "Cp_mass": 2222.0,
        "Cp_molar": 35.64761266,
        "k": 0.0332,
        "H": -7.49e7
    }
}