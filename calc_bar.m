function [u_bar, PI, f_bar] = calc_bar(U_DNS, f_store, NX, NY)
Lx=100;

kx_bar = [0:NY/2 -NY/2+1:-1]'*2.0*pi/Lx;

full_f = f_store;

f_bar = filter_bar(full_f,NX,NX/NY);

full_u = U_DNS;

u_bar = filter_bar(full_u,NX,NX/NY);

uu_full = full_u.*full_u;

uu_coarse = filter_bar(uu_full,NX,NX/NY);

tau = .5*(uu_coarse - u_bar.*u_bar);

fft_PI = 1i*kx_bar.*fft(tau);

PI = real(ifft(fft_PI));
end
