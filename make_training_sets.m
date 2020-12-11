load('DNS_Burgers_s_20.mat')
load('DNS_Force_LES_s_20.mat')

s = 20;

% size of dataset
num_in_set = 1250000;

%shift between start of datasets
set_size = 250000;

for i = 18
    [u_bar, PI, f_bar] = calc_bar(U_DNS(:,(i-1)*set_size+1:((i-1)*set_size)+num_in_set),...
        f_store(:,(i-1)*(set_size/s)+1:((i-1)*(set_size/s)+(num_in_set/s))),1024,128);
    
    save(['./u_bar_region_' num2str(i)  '.mat'],'u_bar')
    save(['./f_bar_region_' num2str(i) '.mat'],'f_bar')
    save(['./PI_region_' num2str(i) '.mat'],'PI')
end
