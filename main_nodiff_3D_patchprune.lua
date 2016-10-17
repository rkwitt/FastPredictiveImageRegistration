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

-- parameters
use_gpu = true
batch_size = 50
epochs = 10
patch_size = 15
scratch = true

config= {
	learningRate = -0.0001,
	momentumDecay = 0.1,
	updateDecay = 0.01
}

-- load training dataset
image_appear_trainset = data_provider.create('data/train_appearance_3D_atlas.t7')
atlas_appear = data_provider.create('data/atlas_appearance_3D_atlas.t7')
image_appear_trainset_map = image_appear_trainset.images:clone():gt(0):float()
atlas_appear_map = atlas_appear.images:clone():gt(0):float()

-- load testing dataset
m0_trainset = data_provider.create('data/train_m0_3D_atlas.t7')

input_batch = torch.Tensor(batch_size, 2, patch_size, patch_size, patch_size)

target_batch = torch.Tensor(batch_size, 3, patch_size, patch_size, patch_size)

total_batch_number = image_appear_trainset.n_images * (image_appear_trainset.images:size(2) - patch_size+1) * (image_appear_trainset.images:size(3) - patch_size+1) * (image_appear_trainset.images:size(4) - patch_size+1)

batch_per_image = (image_appear_trainset.images:size(2) - patch_size+1) * (image_appear_trainset.images:size(3) - patch_size+1) * (image_appear_trainset.images:size(4) - patch_size+1);

batch_per_xy_slice = (image_appear_trainset.images:size(2) - patch_size+1) * (image_appear_trainset.images:size(3) - patch_size+1);


-- build network model
model = create_model.VAE_deformation_parallel_small_3D_noDropout(2, patch_size, 0, 128)
model:training()
--w_init.nn(model, 'xavier_caffe')
criterion = nn.AbsCriterion()
criterion.sizeAverage = false
model = model:cuda()
criterion = criterion:cuda()
atlas_appear.images = atlas_appear.images:cuda();
input_batch = input_batch:cuda()

target_batch = target_batch:cuda()

-- retrieve parameters and gradients
-- it is view, so something like pointer.
parameters, gradients = model:getParameters()
print(parameters:max())
print(parameters:min())

if scratch == false then 
    print("Loading old weights!")

    p = torch.load('snapshots/3D_model_noDropout.t7'):cuda()
    parameters:copy(p)
else
    state = {}
end


-- function to calculate the index of the patch
calculatePatchIdx = function(total_batch_number, batch_per_image, batch_per_xy_slice, num_image, step_x, step_y, step_z, size_x, size_y, size_z, patch_x, patch_y, patch_z)
	-- calculate idx for each xy slice
	local patch_idx = torch.Tensor(1, 1);
	patch_idx[1] = 1;
	local idx_x = 1;
	local idx_y = 1+step_y;
	local cur_idx = (idx_x-1) * (size_y-patch_y+1) + idx_y;
	while (cur_idx <= batch_per_xy_slice) do
		patch_idx = torch.cat(patch_idx, torch.ones(1, 1) * cur_idx, 1);
		-- find the next index
		if (idx_y + step_y <= size_y - patch_y + 1) then
			idx_y = idx_y + step_y
		elseif(idx_y ~= size_y - patch_y + 1) then
			idx_y = size_y - patch_y + 1;
		else
			idx_y = 1
			if ((idx_x ~= (size_x - patch_x + 1)) and ((idx_x + step_x) > (size_x - patch_x + 1))) then
				idx_x = size_x - patch_x + 1;
			else
				idx_x = idx_x + step_x;
			end
		end
		cur_idx = (idx_x-1) * (size_y-patch_y+1) + idx_y;
	end
	patch_idx_xy_slice = patch_idx:clone();

	-- calculate idx for z slices
	local n_z = torch.floor((size_z-patch_z+1) / step_z);
	local len_z = 0;
	if ((size_z-patch_z+1) % step_z == 0) then
		len_z = n_z;
	else
		len_z = n_z+1;
	end

	idx_z = torch.Tensor(1, len_z):squeeze()
    for i_z = 1, n_z do
		idx_z[i_z] = (i_z-1)*step_z+1;
	end

	if (len_z > n_z) then
		idx_z[len_z] = size_z-patch_z+1;
	end

	patch_single_idx = patch_idx_xy_slice:clone();

	for i = 2, len_z do
		patch_single_idx = torch.cat(patch_single_idx, patch_idx_xy_slice:clone() + batch_per_xy_slice * (idx_z[i]-1), 1);
	end

	patch_idx_all = patch_single_idx:clone();
	for i = 1, num_image-1 do
		    collectgarbage()
		patch_idx_all = torch.cat(patch_idx_all, patch_single_idx:clone() + batch_per_image * i, 1);
	end

	return patch_idx_all;
end


--Optimization function
opfunc = function(x)
	collectgarbage()
	if x ~= parameters then
		parameters:copy(x)
	end
	model:zeroGradParameters()

	output = model:forward(input_batch)


	local err = -1* criterion:forward(output, target_batch)
	local df_dw = criterion:backward(output, target_batch):mul(-1)

	model:backward(input_batch,df_dw)

	print('err', err/batch_size)

	return err/batch_size, gradients 
end

for epoch=1,epochs do 
    local lowerbound = 0
    local time = sys.clock()
    --local N = total_batch_number/batch_size/skip3D;
    patch_idx = calculatePatchIdx(total_batch_number, batch_per_image, batch_per_xy_slice, image_appear_trainset.n_images, 14, 14, 14, 128, 128, 128, 15, 15, 15);
	print(patch_idx:size())
	-- find out the background patches and remove them
	patch_idx_select = torch.zeros(patch_idx:size());
	patch_idx_select_length = 0;
	for i = 1, patch_idx:size()[1] do
		slice_idx = patch_idx[i]:squeeze();
		image_number = torch.floor((slice_idx-1) / batch_per_image)+1
		if (torch.floor((slice_idx-1) / batch_per_image) > (slice_idx-1) / batch_per_image) then
			image_number = image_number - 1;
		end

		patch_location = (slice_idx-1) % batch_per_image
		patch_height = torch.floor(patch_location / batch_per_xy_slice) + 1;
		patch_location_xy = patch_location % batch_per_xy_slice;
		patch_row = torch.floor(patch_location_xy / (image_appear_trainset.images:size(2)-patch_size+1))+1;
		patch_column = patch_location_xy % (image_appear_trainset.images:size(2)-patch_size+1)+1;
		image_appear_map = image_appear_trainset_map[{{image_number}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}, {patch_height, patch_height+patch_size-1}}]:clone():reshape(patch_size, patch_size, patch_size)		
		atlas_patch_map = atlas_appear_map[{{patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}, {patch_height, patch_height+patch_size-1}}]:clone():reshape(patch_size, patch_size, patch_size)
		if (torch.sum(image_appear_map) + torch.sum(atlas_patch_map)) > 0 then
			patch_idx_select_length = patch_idx_select_length+1;
			patch_idx_select[i] = 1;
		end
	end

	patch_idx_prune = torch.zeros(patch_idx_select_length, 1);
	location = 1;

	for i = 1, patch_idx:size()[1] do
		if(patch_idx_select[i]:squeeze() == 1) then
			patch_idx_prune[location] = patch_idx[i];
			location = location+1;
		end
	end
	print(patch_idx_prune:size());
	patch_idx = patch_idx_prune:clone();

	local N = patch_idx:size()[1] / batch_size;
	for iter=1,N do
		--Prepare Batch
		for k=1,batch_size do
			i = torch.random()%patch_idx:size()[1];
			slice_idx = patch_idx[i+1]:squeeze();
			image_number = torch.floor((slice_idx-1) / batch_per_image)+1
			if (torch.floor((slice_idx-1) / batch_per_image) > (slice_idx-1) / batch_per_image) then
				image_number = image_number - 1;
			end

			patch_location = (slice_idx-1) % batch_per_image
			patch_height = torch.floor(patch_location / batch_per_xy_slice) + 1;
			patch_location_xy = patch_location % batch_per_xy_slice;
			patch_row = torch.floor(patch_location_xy / (image_appear_trainset.images:size(2)-patch_size+1))+1;
			patch_column = patch_location_xy % (image_appear_trainset.images:size(2)-patch_size+1)+1;

			input_batch[k][1] = image_appear_trainset.images[{{image_number}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}, {patch_height, patch_height+patch_size-1}}]:clone():cuda();
				
			input_batch[k][2] = atlas_appear.images[{{patch_row, patch_row+patch_size-1},{patch_column, patch_column+patch_size-1}, {patch_height, patch_height+patch_size-1}}]:clone():cuda();
				
			target_batch[k] = m0_trainset.images[{{image_number}, {}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}, {patch_height, patch_height+patch_size-1}}]:clone():cuda();
		end

		x, batchlowerbound = rmsprop(opfunc, parameters, config, state)
    	print("epoch: " .. epoch .. "iter: " .. iter .. " Lowerbound: " .. batchlowerbound[1])
    	if iter % 100 == 0 then
    		torch.save('snapshots/3D_model_noDropout.t7', parameters:float())
    	end
	end

end

