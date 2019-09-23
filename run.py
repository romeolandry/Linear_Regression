import DatenGenerierung

daten = DatenGenerierung.DatenGenerierung(4,10)
input_val,output = daten.gen_daten()
daten.data_visualisation(input_val,output)