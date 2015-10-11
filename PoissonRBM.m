function rbm = PoissonRBM(rbm, x, opts)
%%%%%%%%%
% Possion RBM 
% Input : word count 
% Hidden: binary 
% Modified from DeepLearnToolBox.
% Modified by Kai Tian
% (tiank311@gmail.com)
%%%%%%%%%
    assert(isinteger(x), 'x must be a float');			%Judgement: x must be integers, they're word count
    m = size(x, 1);										%number of samples
    numbatches = m / opts.batchsize;					
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
	%%%%%%% Do Iteration %%%%%%%%%%%%%%
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        for l = 1 : numbatches
			% Set batch data
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            %CD-k, as k = 1, so we don't use a loop block
            v1 = batch;			%v1 is observed data
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W'); %Sampling h1 from sigmoid
            v2 = poissrnd(exp(repmat(rbm.b',opts.batchsize,1) + h1 * rbm.W)); %Just one different operation
			%v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
            h2 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');
            %c1(i,j)=p(hi=1|v1)*v1j
            %c2(i,j)=p(hi=1|v2)*v2j
            c1 = h1' * v1;
            c2 = h2' * v2;

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end

function x = rbmdown(rbm, x)
    x = sigm(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
end

function x = rbmup(rbm, x)
    x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
end
