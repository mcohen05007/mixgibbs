%% Example Script for Mixture or Normals Gibbs Sampler.   
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated, please cite as:
% Cohen, M. A. (2015). Gibbs Sampler for Mixture of Normals Model [Computer software]. 
% Retrieved from http://www.macohen.net/software or https://github.com/mcohen05007/mixgibbs
clear
clc

%% 
% Seed Random number geenrator and use the new SIMD-oriented Fast Mersenne
% Twister only for use with MATLAB 2015a or newer
% rng(0,'simdTwister')
rng(100,'twister')

%% Generate Three Component Bivariate Mix-Normal Data

ncomp = 3;
mu = [1 2 3;0 -1 -2]';
sig = zeros(2,2,ncomp);
sig(:,:,1)=[1 .5;.5 1];
sig(:,:,2)=[1 .5;.5 1];
sig(:,:,3)=[1 .5;.5 1];

p = (1:ncomp)/sum(1:ncomp);
n = 1e3;
nvar = size(mu,2);
ind = randsample(ncomp,n,true,p);
y = mvnrnd(mu(ind,:),sig(:,:,ind));

%% Prepare Sampler
% Priors
ncomp = 10; % number of componenets specified
Bbar = zeros(1,nvar);
A = 0.01*eye(1);
nu = 3;
V = nu*eye(nvar);
a = repmat(2,1,ncomp);

% Starting Values
p0 = repmat(1/ncomp,1,ncomp);
z0 =  mnrnd(1,p0,n);

% Allocate Space
R = 5e3; % Number of Draws
keep = 1; % Thinning parameter to reduce autocorrelation
pdraw = zeros(R,ncomp); % component probabilities
compdraw = cell(R,1);   % object to save component mean and variance parameters

%% Run Sampler
tic
for rep = 1:R    
   out = rmixGibbs(y,Bbar,A,nu,V,a,p0,z0);
   p0 = out.p;
   z0 = out.z;
    %% Store draws
    if (mod(rep,keep) == 0) 
        mkeep = rep/keep;
        pdraw(mkeep,:,:) = out.p;
        compdraw{mkeep} = struct('mu',out.mu,'cov',out.cov);
    end
    %% Compute remaining time
    if (mod(rep,100) == 0)
        timetoend = (toc/rep) * (R + 1 - rep);
        hours = floor(timetoend/60/60);
        mins = floor((timetoend/60)-hours*60);
        secs = floor(((timetoend/60)-floor(timetoend/60))*60);
        disp([ '    ', num2str(rep), '          ',num2str(hours),' ', 'Hours',' ',num2str(mins),' ', 'Minutes',' ',num2str(secs),' ','Seconds'])   
    end  
end

%% Evaluate Density Graphically
burnin = 1e4;
ygrid = -4:1:6;
[x1,x2] = meshgrid(ygrid,ygrid);
dengrid = zeros(1,length(ygrid));
for i = 1:length(ygrid)
    for j = 1:length(ygrid)
        den = 0;
        for rep = 1:size(pdraw,1)  
            den = den + pdraw(rep,:)*mvnpdf([ygrid(i) ygrid(j)],compdraw{rep}.mu,compdraw{rep}.cov); 
        end
        dengrid(i,j) = den/R; 
    end
    disp([num2str(round(100*i/length(ygrid))),'%'])
end
surf(x1,x2,dengrid)
