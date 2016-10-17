require 'torch';
require 'nn';
require 'cunn'
require 'gnuplot';
require 'optim';
require 'image';
require 'io';
require 'rmsprop';
create_model = require 'create_model'
w_init = require 'weight-init'
data_provider = require 'data_provider'
matio = require 'matio' -- needed to save the data to .mat file

torch.setdefaulttensortype('torch.FloatTensor')

testset_appearance = data_provider.create('data/test_atlas_appearance.t7')
atlas_appearance = data_provider.create('data/atlas_atlas_appearance.t7')

testset_m0 = data_provider.create('data/test_atlas_m0.t7')

test_apperance_map = testset_appearance.images:clone():gt(0):float()
atlas_appearance_map = atlas_appearance.images:clone():gt(0):float()


patch_size = 15
batch_size = 2000

model = create_model.VAE_deformation_parallel_small_noDropout(2, patch_size, 0, 128)
model = model:cuda()
model:training() -- use training mode to keep Dropout layers

-- retrieve parameters and gradients
-- it is view, so something like pointer.
parameters, gradients = model:getParameters()
p = torch.load('snapshots/2D_model_noDropout.t7'):cuda()
parameters:copy(p);


encoder_outputs={}
lowerbound = 0

total_batch_number = testset_appearance.n_images * (testset_appearance.images:size(2) - patch_size+1) * (testset_appearance.images:size(3) - patch_size+1)
batch_per_image = (testset_appearance.images:size(2) - patch_size+1) * (testset_appearance.images:size(3) - patch_size+1);

test_m0_recon_all = torch.zeros(1, 50, 2, 128, 128);

testset_appearance.images = testset_appearance.images:cuda()
test_m0_recon_all = test_m0_recon_all:cuda()


-- function to calculate the patch location given the sliding window stride
calculatePatchIdx = function(total_batch_number, batch_per_image, num_image, step_x, step_y, size_x, size_y, patch_x, patch_y)

	local patch_idx = torch.Tensor(1, 1);
	patch_idx[1] = 1;
	local idx_x = 1;
	local idx_y = 1+step_y;
	local cur_idx = (idx_x-1) * (size_y-patch_y+1) + idx_y;
	while (cur_idx <= batch_per_image) do
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
	patch_idx_all = patch_idx:clone();

	print(patch_idx[-1])

	for i = 1, num_image-1 do
		patch_idx_all = torch.cat(patch_idx_all, patch_idx:clone() + batch_per_image * i, 1);
	end

	return patch_idx_all;
end

for iter = 1, 1 do
	collectgarbage()
	test_m0_recon = torch.zeros(testset_m0.images:size());
	test_m0_weight = torch.zeros(testset_m0.images:size());
	test_m0_recon = test_m0_recon:cuda()
	test_m0_weight = test_m0_weight:cuda()
	batch = torch.zeros(batch_size, 2, patch_size, patch_size):cuda();

	-- compute the index of the training patches (upper left corner of a patch)
	patch_idx = calculatePatchIdx(total_batch_number, batch_per_image, testset_appearance.n_images, 14, 14, 128, 128, 15, 15);

	print(patch_idx:size())

	-- remove the patches from the background given the patch index vector
	patch_idx_select = torch.zeros(patch_idx:size()); -- 0/1 map indicating whether a patch is kept.
	patch_idx_select_length = 0;
	for i = 1, patch_idx:size()[1] do
		slice_idx = patch_idx[i]:squeeze();
		image_number = torch.floor((slice_idx-1) / batch_per_image)+1
		patch_location = (slice_idx-1) % batch_per_image
		patch_row = torch.floor(patch_location / (testset_appearance.images:size(2)-patch_size+1))+1;
		patch_column = patch_location % (testset_appearance.images:size(2)-patch_size+1)+1;
		test_patch_map = test_apperance_map[{{image_number}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}]:clone():reshape(patch_size, patch_size)
		atlas_patch_map = atlas_appearance_map[{{patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}]:clone():reshape(patch_size, patch_size)
		if (torch.sum(test_patch_map) + torch.sum(atlas_patch_map)) > 0 then
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

	-- start prediction
	full_idx = 1
	while full_idx <= patch_idx:size()[1] do
		-- dynamically change batch size
		current_batch_size = batch_size;
		if (patch_idx:size()[1] - full_idx+1) < batch_size then
			print('change!')
			current_batch_size = patch_idx:size()[1] - full_idx + 1;
			batch = torch.zeros(current_batch_size, 2, patch_size, patch_size):cuda();
		end


		for idx = 1, current_batch_size do
			slice_idx = patch_idx[idx+full_idx-1]:squeeze();
			image_number = torch.floor((slice_idx-1) / batch_per_image)+1
			patch_location = (slice_idx-1) % batch_per_image
			patch_row = torch.floor(patch_location / (testset_appearance.images:size(2)-patch_size+1))+1;
			patch_column = patch_location % (testset_appearance.images:size(2)-patch_size+1)+1;
			input_patch = testset_appearance.images[{{image_number}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}]:clone():reshape(1, 1, patch_size, patch_size)
			batch[idx][1] = input_patch:squeeze();
			atlas_patch = atlas_appearance.images[{{patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}]:clone():reshape(1, 1, patch_size, patch_size)
			batch[idx][2] = atlas_patch:squeeze();

		end

		-- predict initial momentum using the network
		output = model:forward(batch):squeeze();

		-- put the predicted momentum patches together
		for idx = 1, current_batch_size do
			output_slice = output[idx]:clone():squeeze()
			slice_idx = patch_idx[idx+full_idx-1]:squeeze();
			print(slice_idx)
			image_number = torch.floor((slice_idx-1) / batch_per_image)+1
			patch_location = (slice_idx-1) % batch_per_image
			patch_row = torch.floor(patch_location / (testset_appearance.images:size(2)-patch_size+1))+1;
			patch_column = patch_location % (testset_appearance.images:size(2)-patch_size+1)+1;
			print(image_number)
			print(patch_row)
			print(patch_column)
			test_m0_recon[{{image_number}, {}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}] = test_m0_recon[{{image_number}, {}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}] + output_slice;
			test_m0_weight[{{image_number}, {}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}] = test_m0_weight[{{image_number}, {}, {patch_row, patch_row+patch_size-1}, {patch_column, patch_column+patch_size-1}}] + 1;

		end
		full_idx = full_idx + batch_size;
	end
	print(iter)
	print(full_idx)
	test_m0_recon:cdiv(test_m0_weight);
	test_m0_recon_all[iter] = test_m0_recon

end

test_m0_recon_all[test_m0_recon_all:ne(test_m0_recon_all)] = 0		

matio.save('2D_output.mat', {test_m0_recon_all = test_m0_recon_all:float():squeeze()});
