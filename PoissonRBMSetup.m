function rbm = PoissonRBMSetup(arch,opts)
assert(length(arch==2),'Architecture should equal to 2.');
rbm.alpha    = opts.alpha;
rbm.momentum = opts.momentum;

rbm.W  = zeros(arch(2), arch(1));
rbm.vW = zeros(arch(2), arch(1));

rbm.b  = zeros(arch(1), 1);
rbm.vb = zeros(arch(1), 1);
rbm.c  = zeros(arch(2), 1);
rbm.vc = zeros(arch(2), 1);

end