function out = rmixGibbs(y,Bbar,A,nu,V,a,p,z)
% RMIXGIBBS Single draw from posterior of mixture of normals probabiliy model
%   OUT = rmixGibbs(y,Bbar,A,nu,V,a,p,z) samples from the posterior of a
%   mixture of normals posterior probability model.    

%%    
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated, please cite as:
% Cohen, M. A. (2015). Gibbs Sampler for Mixture of Normals Model [Computer software]. 
% Retrieved from http://www.macohen.net/software or https://github.com/mcohen05007/mixgibbs

%%
ncomp = numel(p);
nvar = size(y,2);
if nvar>1
    mu = zeros(ncomp,nvar);
    cov = zeros(nvar,nvar,ncomp);
    fi = zeros(nvar,nvar,ncomp);
else
    mu = zeros(ncomp,1);
    cov = zeros(ncomp,1);  
    fi = zeros(ncomp,1); 
end
for k = 1:ncomp
    yk = y(z(:,k)==1,:);
    nv = size(yk,1);
    Data = struct('Y',yk,'X',ones(nv,1));
    out = bmreg(Data,Bbar,A,nu,V,nv,1,nvar);
    if nvar>1
        mu(k,:) = out.beta';
        cov(:,:,k) = out.Sigma;
        fi(:,:,k) = out.FI;
    else
        mu(k) = out.beta;
        cov(k) = out.Sigma;
        fi(k) = out.FI;
    end   
end
ptil = comprob(y,mu,cov,p);
z = mnrnd(1,ptil);
p = dirichrnd(a + sum(z));
out = struct('mu',mu,'cov',cov,'fi',fi,'p',p,'z',z);
end

%% Supporting Functions

%% Multiple Regression
function out = bmreg(Data,Bbar,A,nu,V,T,k,neq) 
% BMREG Single draw from posterior of multivariate-error Bayesian regression model
%   OUT = bmreg(Data,Bbar,A,nu,V,T,k,neq) samples from the posterior on a
%   hierarchical multiple regression model. The structure Data contains .
%   The structure Prior contains. The structure Mcmc contaions.      

%%    
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated, please cite as:
% Cohen, M. A. (2015). Gibbs Sampler for Mixture of Normals Model [Computer software]. 
% Retrieved from http://www.macohen.net/software or https://github.com/mcohen05007/mixgibbs

%%
RA = chol(A);
W = [Data.X;RA];
Z = [Data.Y;RA*Bbar];
IR = eye(k)/chol(W'*W);
Btilde = (IR*IR')*W'*Z;
res = Z-W*Btilde;
wdraw = rwishart(nu+T,eye(neq)/(res'*res + V));
beta = Btilde + IR*randn(k,neq)*wdraw.CI';
out = struct('beta',beta,'Sigma',wdraw.IW,'FI',wdraw.W);
end

%% Wishart
function rout=rwishart(nu, V) 
% RWISHART random draw from wishart distribution
%   rout = rwishart(nu,V) give wishart location and scale parameters nu and
%   V this function gives back a structure called rout that contains a
%   wishart draw, its inverse, and their respective cholesky roots
%

%%
m = size(V,1);
if (m > 1)
    T = diag(sqrt(chi2rnd(nu,m,1)));
    T = tril(ones(m,m),-1).* randn(m) + T;
else 
    T = sqrt(chi2rnd(nu));
end
U = chol(V);
C = T' * U;
CI = eye(m)/C;
W = C'*C;
IW = CI*CI';
rout = struct('C',C,'CI',CI,'W', W,'IW',IW);
end

%% Dirichlet
function x = dirichrnd(a)
%%
y = gamrnd(a,1);
x = y/sum(y);
end

%% Combine Dirichlet Probabilities with Nornal Components
function ptil = comprob(y,mu,cov,p)
%%
ncomp = numel(p);
n = size(y,1);
phi = zeros(n,ncomp);
for k = 1:ncomp
    if size(y,2)>1
        phi(:,k) = mvnpdf(y,mu(k,:),cov(:,:,k));
    else
        phi(:,k) = normpdf(y,mu(k),sqrt(cov(k)));
    end
end
ptil = (phi.*(ones(n,1)*p))./((phi*p')*ones(1,ncomp));
end