
################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)

for iteration in range(1,opt.niter+1):

    optimizer.zero_grad()
    try:
        style,_ = style_loader.next()
    except IOError:
        style,_ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style,_ = style_loader.next()
    except:
        continue
    styleV.resize_(style.size()).copy_(style)
    
    for step, (frames,path0) in enumerate(content_loader):
        frames_done = list()
        print("path0",path0)
        ssim_0 = []
        ssim_1 = []
        frames_fea = []
        l = len(frames)
        for i in range(0,l):
            x_t = frames[i]
            if(i == 0):
                fx_t = vgg(x_t.cuda().squeeze(1))
                fs = vgg(styleV)
                if(opt.layer == 'r41'):
                    feature,transmatrix = matrix(fx_t[opt.layer],fs[opt.layer])
                else:
                    feature,transmatrix = matrix(fx_t,fs)
                frames_fea = feature
                transfer = dec(feature)
                fx_tt = transfer
                frames_done.append(fx_tt)
            else:
                fx_t = vgg(x_t.cuda().squeeze(1))
                fs = vgg(styleV)
                if(opt.layer == 'r41'):
                    feature,transmatrix = matrix(fx_t[opt.layer],fs[opt.layer])
                else:
                    feature,transmatrix = matrix(fx_t,fs)
                
                f1 = feature.cuda()
                f2 = frames_fea
                out,attention_value = attention(f1,f2)
                frames_fea = out
                transfer = dec(out)
                fx_tt = transfer
                frames_done.append(fx_tt)

            content = frames[l-1].cuda()
            sF_loss = vgg5(styleV)
            cF_loss = vgg5(content)
            tF = vgg5(fx_tt)
        #caculate the ssim array of source video and stylized video
        ssim_0 = np.zeros(l-1)
        ssim_1 = np.zeros(l-1)
        ssim_m0 = np.zeros(l-m)
        ssim_m1 = np.zeros(l-m)
        for i in range(0,l-1):
            x_t = frames[i]
            x_t1 = frames[i+1]
            y_t = frames_done[i]
            y_t1 = frames_done[i+1]
            ssim0 = ssim(x_t,x_t1)
            ssim1 = ssim(y_t,y_t1)
            #convert tensor to numpy
            ssim0 = ssim0.cpu()
            ssim1 = ssim1.cpu()
            ssim0 = ssim0.detach().numpy()
            ssim1 = ssim1.detach().numpy()
            ssim_0[i] = ssim0
            ssim_1[i] = ssim1
            ssim_0 = np.array(ssim_0)
            ssim_1 = np.array(ssim_1)
            if (i+m<l):
                x_tm = frames[i+m]
                y_tm = frames_done[i+m]
                ssimm0 = ssim(x_t,x_tm)
                ssimm1 = ssim(y_t,y_tm)
                ssimm0 = ssimm0.cpu()
                ssimm1 = ssimm1.cpu()
                ssimm0 = ssimm0.detach().numpy()
                ssimm1 = ssimm1.detach().numpy()
                ssim_m0[i] = ssimm0
                ssim_m1[i] = ssimm1
                ssim_m0 = np.array(ssim_m0)
                ssim_m1 = np.array(ssim_m1)

        # caculate the loss 
        ssim_loss_torch = ssim_loss(ssim_1,ssim_0)
        ssim_loss_torch.float()
        ssim_loss_long_torch = ssim_loss(ssim_m1,ssim_m0)
        ssim_loss_long_torch.float()
        loss,styleLoss,contentLoss,ssim_loss_torch, ssim_loss_long_torch = criterion(tF,sF_loss,cF_loss,ssim_loss_torch.float().cuda(),ssim_loss_long_torch.float().cuda())
        loss.backward(retain_graph=True)
        optimizer.step()

            
        print('Iteration: [%d/%d] Loss: %.4f contentLoss: %.4f styleLoss: %.4f ssim_loss: %.4f ssim_loss_l: %.4f Learng Rate is %.8f'%#
            (opt.niter,iteration,loss,contentLoss,styleLoss,ssim_loss_torch,ssim_loss_long_torch,optimizer.param_groups[0]['lr']))#
            

        adjust_learning_rate(optimizer,iteration)

        if(iteration > 0 and (iteration) % opt.save_interval == 0):
            torch.save(matrix.state_dict(), '%s/%s' % (opt.outf,opt.layer))    
