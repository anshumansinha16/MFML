%f1 = @(z) sin(12*(z+0.2))./(z+0.2);
M = 200; % Replace with the desired number of data points
T = sort(rand(M, 1));


f0 = @(z) sin(12*(z+0.2))./(z+0.2);
y = f0(T) + 0.1*randn(M, 1);
kern = @(z1,z2,q) exp(-(z1-z2).^2/(2*q^2));
i = 0;

sig = 1/1000;

delta = .004;
K = kern(T, T', sig);
w_hat = inv(K + delta*eye(M))*y;
f_hat = @(z) kern(z,T',sig)*w_hat;
samperr = norm(f_hat(T)-y);
func = @(z) (f_hat(z)-f0(z)).^2;
generr = (integral(func,0,1,'ArrayValued',true)).^0.5;
figure(1)
cla
plot(T,y,'o')
hold on
s = linspace(0,1,1000)';
plot(s,f_hat(s),'r')
plot(s,f0(s),'g')
title(['Sigma=',num2str(sig),'; SampErr=',num2str(samperr),'; GenErr=',num2str(generr)])
legend('Sample Values','Fitted Function','True Function')
saveas(gcf,['sin_approx_d',num2str(i),'.png'])
i = i+1;