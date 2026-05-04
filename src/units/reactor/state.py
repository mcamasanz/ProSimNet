# src/units/reactor/state.py
# pack_state_vector / unpack_state_vector — pendiente de implementar
# Layout sv (a definir tras decisiones de diseño):
#   Sin shell_tube:  [C(nc,N), Hg(N), [Ts(N)]]        tamaño = (nc+1[+1])·N
#   Con shell_tube:  [C(nc,N), Hg(N), [Ts(N)], Tw(N)] tamaño = (nc+1[+1]+1)·N
# Ts solo presente si catalyst_config is not None (modo lecho).
