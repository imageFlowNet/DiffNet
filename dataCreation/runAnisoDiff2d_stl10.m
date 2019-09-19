%% test program to demonstrate the AnisoDiff2d function.

%Load stl-10 files in unlabeled.mat, download from https://cs.stanford.edu/~acoates/stl10/
load('PATH2STL10/unlabeled.mat')

N = 96; % image size. Assume square;
N3=N;
diffIm = zeros(N3,N3,4,100000,'single');
imagesTrue = zeros(N3,N3,4,100000,'single');
%%

for iii=1:100000


    im = double(rgb2gray(reshape(X(iii,:),[N N 3])));
    im = im./max(im(:)); % set max to 1 for convenience

    imagesTrue(:,:,iii)=im;

    ADopts.ADtype = 'NonLinear';
    ADopts.ADiter = 4;
    ADopts.ADdeltaT = 0.1;
    % ADopts.ADdelta = 4; % with this time step, implicit solver is needed
    ADopts.ADsolver = 'Explicit'; % implicit is stable, but a bit slow;

    kapPM1 = @(s,T) ((T^2)./(s.^2 + T^2)); % note that threshold is set from ADthresh.
    ADopts.ADthresh =  0.2; 
    ADopts.ADkapfunc =  kapPM1;

    [imout_PM1,KPM1,kapim] = AnisoDiff2d(im,ADopts);

    %  to display diffusivity, take -diag of graph/4

    diffIm(:,:,iii)=imout_PM1;


    if mod(iii,100)==0
        display(iii)
    end

end