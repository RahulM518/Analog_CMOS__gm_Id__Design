# Analog_CMOS__gm_Id__Design
Automation of gm/Id design using Python scripts on techplots csvs from Cadence Virtuoso\
What is done so far:
1) gm/Id based graphs for NMOS and PMOS
2) extraction of parameters like Id/W, L and ft via simple interpolation
3) Obtaining reverse values of gmro and gm/Id
4) Given an L and gm/Id, find effective gmro,id/W or ft values depending on graph
# Results:
1) Over 98% accuracy in designs for both 5T OTA and LDO Design (Internal Compensated Miller) (1.4V-1V)
2) Performed for 180nm, gpdk090 and gpdk045 pdks based scripting for observing effective resultant values and reverse parameter evaluation
