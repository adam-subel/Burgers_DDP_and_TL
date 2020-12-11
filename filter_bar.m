function var_bar = filter_bar(u, N, n_sub)
var_bar=zeros(N/n_sub,length(u));

for i=1:(N/n_sub)
    var_bar(i,:)=mean(u(n_sub*(i-1)+1:n_sub*(i-1)+n_sub,:));
end
end