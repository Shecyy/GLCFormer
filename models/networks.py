import models.archs.GLformer as GLformer


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    if which_model == 'gl_former':
        netG = GLformer.GLformer(opt=opt)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
