reset
set term epslatex color solid
set output "linfit.eps"

f(x)=m*x+b
set xlabel "Spannung U [V]"
set ylabel "Stromstärke I [A]"

fit f(x) 'TD2_RT.txt' via m,b
p  'TD2_RT.txt' t "Messdaten", f(x) t "fit"

set output
