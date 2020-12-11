% This is to compute dataset for analysis

load('DNS_Burgers_s_20_10_mil.mat')
load('DNS_Force_LES_s_20_10_mil.mat')


[u_bar,PI,f_bar]=calc_bar(U_DNS(:,1:20:end),f_store,1024,128);

save('PI_all_regions.mat','PI')
save('u_bar_all_regions.mat','u_bar')
save('f_bar_all_regions.mat','f_bar')



