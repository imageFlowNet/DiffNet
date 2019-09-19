function [imout,Klap,kapim ] = AnisoDiff2d(imin,options)

% Assuming im is a NxN image, this function runs different diffusion
% operators on im to prodice imout
% options :
%   ADK = Klap;
%       Klap is (sparse) N^2 x N^2 matrix created externally
%   ADtype = 'Iso'/'Fixed'/'EdgeWeighted'/'NonLinear';
%      'Iso' means use a fixed isotropic Laplacian
%      'Fixed' means that the diffusion operator is passed in (option ADK)
%      'EdgeWeighted' means operator is calculated based on initial image
%      'NonLinear' means operator is calculated during iterations
%   ADkapfunc = func;
%      User supplied function that generates kappa image from image
%      gradient. Used for ADtype = 'EdgeWeighted' or 'NonLinear'
%   ADiter = integer
%      number of iterations
%   ADdeltaT = float
%      time step of each iteration
%   ADsolver = 'Explicit','Implicit','SemiImplicit'
%      method for solving the diffusion step
%
%
%   Written by Simon Arridge, UCL


%% set options and/or default options
im = double(imin); % jsut to be on the safe side...
if(isfield(options, 'ADsolver'))
    ADsolver = options.ADsolver;
else
    ADsolver = 'Explicit';
end

if(isfield(options, 'ADiter'))
    ADiter = options.ADiter;
else
    ADiter = 1;
end

if(isfield(options, 'ADdeltaT'))
    ADdeltaT = options.ADdeltaT;
else
    ADdeltaT = 1;
end

if(isfield(options, 'ADtype'))
    ADtype = options.ADtype;
else
    ADtype = 'Iso';
end
% ADdeltaT
% ADiter
%% create a few useful matrix operators
[n1,n2] = size(im);
N = n1; % assume square.
oneN = ones(N+1,1);
%Dx = sparse(N-1,N);
D1x = spdiags([-oneN oneN],-1:0,N+1,N);
D2x = -D1x'*D1x;
% The following allows interpolation to pixel mid-points;
intx1d = spdiags([0.5*oneN 0.5*oneN],-1:0,N+1,N);

% this form of the Laplacian (D2x) implicitly assumes Dirichlet b.c.s of
% zero on the domain outside the interval [1-N]. I.e. f(0) = f(N+1) = 0

D1x2d = kron(speye(N),D1x);
D1y2d = kron(D1x,speye(N));
intx2d = kron(speye(N),intx1d);
inty2d = kron(intx1d,speye(N));

Lapl = -(D1x2d'*D1x2d + D1y2d'*D1y2d);

%% include the image regularisation functionals.
% first estimate a threshold value
if(isfield(options, 'ADthresh'))
    ADthresh = options.ADthresh;
else
    [gx,gy] = gradient(im);
    gim = sqrt(gx.^2 + gy.^2);
    gmax = max(gim(:));
    ADthresh = 0.1*gmax;
end


kapTK1 = @(s,T) (ones(size(s)));
kapTV =  @(s,T) (1./s);
kapsTV = @(s,T) ((T)./sqrt(s.^2 + T^2));
kapPM1 = @(s,T) ((T^2)./(s.^2 + T^2));

%% perform the diffusion

% default in case of wrong options
imout = im;
Klap = speye(N*N);
BALopts.Verbose = false;
BALopts.thresh = ADthresh; % its clunky passing this around...
switch ADtype
    case 'Iso'
        Klap = Lapl;
        h = reshape(im,[],1);
        switch ADsolver
            case 'Implicit'
                AA = speye(N*N) - ADdeltaT .* Klap;
                for k = 1:ADiter
                    h = AA\h;
                end
            case 'Explicit'
                AA = speye(N*N) + ADdeltaT .* Klap;
                for k = 1:ADiter
                    h = AA*h;
                end
            case 'SemiImplicit'
                disp(['solver ',ADsolver,' not implemented yet']);
            otherwise
                disp(['Unknown solver method ',ADsolver]);
        end
        imout = reshape(h,N,N);
    case 'Fixed'
        if(isfield(options, 'ADK'))
            Klap = options.ADK;
        else
            disp('Diffusion operator is missing with ADtype=Fixed. Revert to isotropic');
            Klap = Lapl;
        end
        h = reshape(im,[],1);
        switch ADsolver
            case 'Implicit'
                AA = speye(N*N) - ADdeltaT .* Klap;
                for k = 1:ADiter
                    h = AA\h;
                end
            case 'Explicit'
                AA = speye(N*N) + ADdeltaT .* Klap;
                for k = 1:ADiter
                    h = AA*h;
                end
            case 'SemiImplicit'
                disp(['solver ',ADsolver,' not implemented yet']);
            otherwise
                disp(['Unknown solver method ',ADsolver]);
        end
        imout = reshape(h,N,N);
    case 'EdgeWeighted'        
        if(isfield(options, 'ADkapfunc'))
            kapfunc = options.ADkapfunc;
        else
            disp('Diffusivity function not supplied in Options.kapfunc. Revert to isotropic')
            kapfunc = kapTK1;
        end
        [Klap,gim,kapim] = BuildAnisoLaplacian(im,kapfunc,N,BALopts);
        h = reshape(im,[],1);

        switch ADsolver
            case 'Implicit'
                AA = speye(N*N) - ADdeltaT .* Klap;
                for k = 1:ADiter
                    h = AA\h;
                end
            case 'Explicit'
                AA = speye(N*N) + ADdeltaT .* Klap;
%                 save AAmat AA
                for k = 1:ADiter
                    h = AA*h;
                end
            case 'SemiImplicit'
                disp(['solver ',ADsolver,' not implemented yet']);
            otherwise
                disp(['Unknown solver method ',ADsolver]);
        end
        imout = reshape(h,N,N);
    case 'NonLinear'
        if(isfield(options, 'ADkapfunc'))
            kapfunc = options.ADkapfunc;
        else
            diap('Diffusivity function not supplied in Options.kapfunc. Revert to isotropic')
            kapfunc = kapTK1;
        end
        h = reshape(im,[],1);
        switch ADsolver
            case 'Implicit'
                for k = 1:ADiter
                    [Klap,gim,kapim] = BuildAnisoLaplacian(reshape(h,N,N),kapfunc,N,BALopts);
                    AA = speye(N*N) - ADdeltaT .* Klap;                    
                    h = AA\h;
                end
            case 'Explicit'
                for k = 1:ADiter
                    [Klap,gim,kapim] = BuildAnisoLaplacian(reshape(h,N,N),kapfunc,N,BALopts);
                    AA = speye(N*N) + ADdeltaT .* Klap;
                    h = AA*h;
                end
            case 'SemiImplicit'
                disp(['solver ',ADsolver,' not implemented yet']);
            otherwise
                disp(['Unknown solver method ',ADsolver]);
        end
        imout = reshape(h,N,N);
    otherwise
        disp(['Unknown diffusion type ', ADtype]);
end
end % end of AnisoDiff2D function
%%
function [aLap,gim,kap2d,kap1x,kap1y] = BuildAnisoLaplacian(im,kapfunc,N,options)

% assume a square NxN image 
% kap is a function that computes NxN diffusivity image from im
% Note that kap might be fixed if not dependent on im
% some options possible...

%[n1,n2] = size(im);
%N = n1; % assume square.
n1 = N; n2 = N;
oneN = ones(N+1,1);
D1x = spdiags([oneN -oneN],-1:0,N+1,N);
% D2x = -D1x'*D1x; 
% The following allows interpolation to pixel mid-points;
intx1d = spdiags([0.5*oneN 0.5*oneN],-1:0,N+1,N);

D1x2d = kron(speye(N),D1x);
D1y2d = kron(D1x,speye(N));
intx2d = kron(speye(N),intx1d);
inty2d = kron(intx1d,speye(N));


h = reshape(im,[],1);
gx1d = D1x2d*h; gx2d = reshape(gx1d,n1+1,n2);
gy1d = D1y2d*h; gy2d = reshape(gy1d,n1,n2+1);
gim = sqrt(gx2d(1:n1,:).^2 + gy2d(:,1:n2).^2);

% we can get an alternative gradient image using interpolation operators.
% gg1d = sqrt((intx2d'*gx1d).^2 + (inty2d'*gy1d).^2);
% gg2d = reshape(gg1d,n1,n2);

% define kappa
kap2d = kapfunc(gim,options.thresh);
if(options.Verbose)
    disp(['N = ',num2str(N)]);
    [nk1,nk2] = size(kap2d);
    disp(['Nk1=',num2str(nk1),' Nk2=',num2str(nk2)]);    
end
kap1d = reshape(kap2d,[],1);

kap1x = spdiags(intx2d*kap1d,0:0,(n1+1)*n2,(n1+1)*n2);
kap1y = spdiags(inty2d*kap1d,0:0,n1*(n2+1),n1*(n2+1));

aLap = -(D1x2d'*kap1x*D1x2d + D1y2d'*kap1y*D1y2d);


end