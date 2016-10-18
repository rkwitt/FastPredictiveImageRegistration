--[[ Traing encoder/decoder on 2D OASIS slices to predict initial momenta for
     LDDMM registration of the atlas image to the source images.

     Author: Xiao Yang, 2016
  ]]


-- imports
require 'torch';
require 'nn';
require 'cunn';
require 'cutorch';
require 'gnuplot';
require 'optim';
require 'image';
require 'io';
require 'rmsprop'


create_model = require 'create_model'
w_init = require 'weight-init'
data_provider = require 'data_provider'
torch.setdefaulttensortype('torch.FloatTensor')


-- parameters (adjust if required)
use_gpu = true -- run on GPU
batch_size = 300 -- use batch sizes of 300 elements
epochs = 10 -- run for 10 epochs
patch_size = 15 -- use patch size 15x15
scratch = true -- run from scratch, otherwise load *old* weights

-- configure optimizer
config= {
  learningRate = 0.0005,
  momentumDecay = 0.1,
  updateDecay = 0.01
}


-- load images (OASIS 2D slices)
image_appear_trainset = data_provider.create('data/train_atlas_appearance.t7')
-- load atlas image
atlas_appear = data_provider.create('data/atlas_atlas_appearance.t7')
-- load initial momenta (i.e., the encoder/decoder target)
m0_trainset = data_provider.create('data/train_atlas_m0.t7')


-- allocate memory (default: 300x2x15x15) for encoder/decoder source
input_batch = torch.Tensor(batch_size, 2, patch_size, patch_size)
-- allocate memory (default: 300x2x15x15) for encoder/decoder target
target_batch = torch.Tensor(batch_size, 2, patch_size, patch_size)
-- calculate the total number of batches
total_batch_number = image_appear_trainset.n_images -- e.g., 100
  * (image_appear_trainset.images:size(2) - patch_size+1) -- e.g., 128-15+1
  * (image_appear_trainset.images:size(3) - patch_size+1) -- e.g., 128-15+1
-- calculate the batch size for one image
batch_per_image = (image_appear_trainset.images:size(2) - patch_size+1) *
  (image_appear_trainset.images:size(3) - patch_size+1)


-- build network model
model = create_model.VAE_deformation_parallel_small_noDropout(2, patch_size, 0, 128)
print(model)
-- w_init.nn(model, 'xavier_caffe')
-- set loss
criterion = nn.AbsCriterion()
criterion.sizeAverage = false


-- transfer to GPU
model = model:cuda()
criterion = criterion:cuda()
image_appear_trainset.images = image_appear_trainset.images:cuda()
atlas_appear.images = atlas_appear.images:cuda();
m0_trainset.images = m0_trainset.images:cuda()
input_batch = input_batch:cuda()
target_batch = target_batch:cuda()


-- retrieve parameters and gradients
parameters, gradients = model:getParameters()


--
if scratch == false then
    print("Loading old weights!")
    p = torch.load('snapshots/parameters_p31_f128_dropout_small_2channel_appearance_m0_new_parallel_fullDropout.t7'):cuda()
    parameters:copy(p)
else
    state = {}
end


-- optimizer function
opfunc = function(x)
  collectgarbage()

  if x ~= parameters then
    parameters:copy(x)
  end
  model:zeroGradParameters()

  -- run forward pass through model
  local output = model:forward(input_batch)
  -- compute loss
  local err = criterion:forward(output, target_batch)
  -- compute the gradients of the loss function associated to the criterion
  local df_dw = criterion:backward(output, target_batch)

  -- perform a backprop step
  model:backward(input_batch,df_dw)

  -- return error normalized by batch size + gradient(s)
  return err/batch_size, gradients
end


print("")
print("Use GPU: " .. tostring(use_gpu))
print("Batch size: " .. batch_size)
print("Epochs: " .. epochs)
print("Patch size: " .. patch_size .. "x" .. patch_size)
print("Run from scratch: " .. tostring(scratch))
print("")


-- training, e.g., run for 10 epochs
for epoch=1,epochs do

    local lowerbound = 0
    local time = sys.clock()
    local N = total_batch_number/batch_size
    --[[
    Note: per default, the total_batch_number = 100*(128-15+1)*(128-15+1)=1299600
    and the batch_size is 15, so N=3000.7575 and we eventually will iterate from
    1 to 3000.
    ]]

    -- run over all batches
    for iter=1,N do
      -- go through all (e.g., 300) elements of a batch
      for k=1,batch_size do
        -- e.g., get a random index in [1,1299600]
        i = torch.random(1,4294847700)%total_batch_number
        -- get the very image from which we get the 15x15 patches
        image_number = torch.floor(i / batch_per_image)+1
        -- get the patch location
        patch_location = i % batch_per_image
        -- get corresponding row
        patch_row = torch.floor(patch_location / (image_appear_trainset.images:size(2)-patch_size+1))+1;
        -- get corresponding column
        patch_column = patch_location % (image_appear_trainset.images:size(2)-patch_size+1)+1;


        -- fill the input patch tensor's rows
        input_batch[k][1] = image_appear_trainset.images[{{image_number}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}]:clone()
        -- fill the input patch tensor's columns
        input_batch[k][2] = atlas_appear.images[{{patch_row, patch_row+patch_size-1},{patch_column, patch_column+patch_size-1}}]:clone()
        -- fill the target: these are the momenta in x and y direction
        target_batch[k] = m0_trainset.images[{{image_number}, {}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}]:clone()
      end


      -- run RMSPROP for NEW parametes and the function evaluated before update
      x, batchlowerbound = rmsprop(opfunc, parameters, config, state)
      -- calculate lower bound
      lowerbound = lowerbound + batchlowerbound[1]


      -- status update per iteration
      print("Epoch: " .. epoch .. "/".. epochs ..
            ", Iteration: " .. iter .. "/" .. N ..
            ", Lowerbound: " .. batchlowerbound[1])
      if iter % 100 == 0 then
        torch.save('snapshots/parameters_p15_f128_dropout_small_2channel_appearance_m0_new_parallel_noDropout_newatlas.t7', parameters:float())
      end
    end

    -- status update per epoch
    print("Epoch: " .. epoch .. "/".. epochs .. ", Lowerbound: " .. lowerbound/N .. ", Time: " .. sys.clock() - time)

    -- keep track of the lowerbound over time
    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end
end
