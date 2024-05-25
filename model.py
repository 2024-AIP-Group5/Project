import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch_scatter import scatter_mean
# from torch_geometric.nn import MetaLayer


TIME_WINDOW = 24
PRED_LEN = 6

class Model(nn.Module):
	def __init__(self, mode, encoder, w_init, w, x_em, date_em, loc_em, edge_h, gnn_h, gnn_layer, city_num, group_num, pred_step, device):
		super(Model, self).__init__()
		self.device = device
		self.mode = mode
		self.encoder = encoder
		self.w_init = w_init
		self.x_em = x_em
		self.city_num = city_num
		self.group_num = group_num
		self.edge_h = edge_h
		self.gnn_layer = gnn_layer
		self.pred_step = pred_step

		if self.encoder == 'self':
			self.encoder_layer = TransformerEncoderLayer(8, nhead=4, dim_feedforward=256)
			# self.x_embed = Lin(8, x_em)
			self.x_embed = Lin(TIME_WINDOW*8, x_em)

		elif self.encoder == 'lstm':
			self.input_LSTM = nn.LSTM(8,x_em,num_layers=1,batch_first=True)


		if self.w_init == 'rand':
			self.w = Parameter(torch.randn(city_num,group_num).to(device,non_blocking=True),requires_grad=True)

		elif self.w_init == 'group':
			self.w = Parameter(w,requires_grad=True)



		self.loc_embed = Lin(2, loc_em)

		self.u_embed1 = nn.Embedding(12, date_em) #month
		self.u_embed2 = nn.Embedding(7, date_em) #week
		self.u_embed3 = nn.Embedding(24, date_em) #hour

		self.edge_inf = Seq(Lin(x_em*2+date_em*3+loc_em*2,edge_h),ReLU(inplace=True))

		self.group_gnn = nn.ModuleList([NodeModel(x_em+loc_em,edge_h,gnn_h)])

		for i in range(self.gnn_layer-1):
			self.group_gnn.append(NodeModel(gnn_h,edge_h,gnn_h))

		self.global_gnn = nn.ModuleList([NodeModel(x_em+gnn_h,1,gnn_h)])

		for i in range(self.gnn_layer-1):
			self.global_gnn.append(NodeModel(gnn_h,1,gnn_h))

		####################################################################################################
		if self.mode == 'feedback':
			self.predMLP = Seq(Lin(gnn_h, 128), ReLU(inplace=True), Lin(128, 128), ReLU(inplace=True), Lin(128, 1), ReLU(inplace=True))
		
		if self.mode == 'temp':
			self.decoder = DecoderModule(x_em,edge_h,gnn_h,gnn_layer,city_num,group_num,device)
			self.predMLP = Seq(Lin(gnn_h,16),ReLU(inplace=True),Lin(16,self.pred_step),ReLU(inplace=True))			
			self.TemporalAggregateMLP = Seq(Lin(gnn_h+8,gnn_h),ReLU(inplace=True))	

		if self.mode == 'both':
			self.predMLP = Seq(Lin(gnn_h, 128), ReLU(inplace=True), Lin(128, 128), ReLU(inplace=True), Lin(128, 1), ReLU(inplace=True))
			self.TemporalAggregateMLP = Seq(Lin(gnn_h+8,gnn_h),ReLU(inplace=True))	

		if self.mode == 'test':
			self.tmp_encoder_layer = TransformerEncoderLayer(32, nhead=4, dim_feedforward=256)
			self.predMLP = Seq(Lin(gnn_h, 128), ReLU(inplace=True), Lin(128, 128), ReLU(inplace=True), Lin(128, 1), ReLU(inplace=True))
		####################################################################################################

		if self.mode == 'ag':
			self.decoder = DecoderModule(x_em,edge_h,gnn_h,gnn_layer,city_num,group_num,device)
			self.predMLP = Seq(Lin(gnn_h,16),ReLU(inplace=True),Lin(16,1),ReLU(inplace=True))

		if self.mode == 'full':
			self.decoder = DecoderModule(x_em,edge_h,gnn_h,gnn_layer,city_num,group_num,device)
			self.predMLP = Seq(Lin(gnn_h,16),ReLU(inplace=True),Lin(16,self.pred_step),ReLU(inplace=True))			

	def batchInput(self, x, edge_w, edge_index):
		# new_x
		sta_num = x.shape[1]
		x = x.reshape(-1, x.shape[-1])

		# edge_w
		edge_w = edge_w.reshape(-1,edge_w.shape[-1])

		# edge_index
		for i in range(edge_index.size(0)):
			edge_index[i,:] = torch.add(edge_index[i,:], i*sta_num)
		edge_index = edge_index.transpose(0,1)
		edge_index = edge_index.reshape(2,-1)

		return x, edge_w, edge_index
	
 

	def forward(self, x, u, edge_index, edge_w, loc):
		# Shape: (batch, 209, 24, 8)
		x = x.reshape(-1,x.shape[2],x.shape[3]) # Shape: (209*batch, 24, 8)

		''' Self Attention '''
		if self.encoder == 'self':
			# [S,B,E]

			x = x.transpose(0,1) # Shape: (24, batch*209, 8)
			x = self.encoder_layer(x) # self-attention, Shape: (24, batch*209, 8)
			x = x.transpose(0,1) # Shape: (batch*209, 24, 8)

			####################################################################################################	
			if self.mode == 'temp' or self.mode == 'both':
				x2 = x.reshape(-1,self.city_num,x.shape[1],x.shape[2]) # Temporal
			####################################################################################################	

			x = x.reshape(-1,self.city_num,TIME_WINDOW*x.shape[-1]) # Shape: (batch, 209, 24*8)
			x = self.x_embed(x) # Linear(24*8, x_em:32) Shape: (batch, 209, 32)
			# x = x.reshape(-1,self.city_num,TIME_WINDOW,x.shape[-1])
			# x = torch.max(x,dim=-2).values
			# print(x.shape)

		elif self.encoder == 'lstm':
			_,(x,_) = self.input_LSTM(x)
			x = x.reshape(-1,self.city_num,x.shape[-1])


		####################################################################################################
		# Temporal
		if self.mode == 'temp' or self.mode == 'both':
			x2 = torch.index_select(x2, dim=2, index=torch.tensor([3, 7, 11, 15, 19, 23]).to('cuda')) ## ( batch size, num of city, selected 6h, 8features )
			x2 = x2.transpose(1,2) ## ( batch size, selected 6h , num of city, 8features ) [batch size, 6, 209, 8]

			h24 = x2[:, -1:, :, :]  # [batch size, 1, 209, 8]
			h_other5 = x2[:, :-1, :, :]  # [batch size, 5, 209, 8]

			attention_scores = torch.einsum('bijk,bljk->bijl', h_other5, h24)  # [batch size, 5, 209, 209]
			attention_weights = F.softmax(attention_scores, dim=1)  # [batch size, 5, 209, 209]

			attention_weighted_sum = torch.einsum('bijk,bijl->bijk', h_other5, attention_weights)  # [batch size, 5, 209, 8]
			x2 = torch.sum(attention_weighted_sum, dim=1)  # [batch size, 209, 8]
		####################################################################################################



		''' Differentiable grouping network 
			City to City Group	'''
		
		# S
		w = F.softmax(self.w, dim=1) # w: (209, group_num:15)
		w1 = w.transpose(0, 1)
		w1 = w1.unsqueeze(dim=0)
		w1 = w1.repeat_interleave(x.size(0), dim=0) # w1: (batch, group_num, 209)

		# city group 
		loc = self.loc_embed(loc) # Linear(2, loc_em:12), shape: (batch, 209, 12)
		x_loc = torch.cat([x,loc],dim=-1) # X, L (batch, 32+12=44)
		g_x = torch.bmm(w1,x_loc) # g_x: (batch, group_num, 44)



		''' Group Correlation Encoding Module 
			Edge Connection '''

		# T
		u_em1 = self.u_embed1(u[:,0]) # Embedding(12, date_em=4) Shape: (batch, 209, 4)
		u_em2 = self.u_embed2(u[:,1]) # Embedding(7, date_em=4)
		u_em3 = self.u_embed3(u[:,2]) # Embedding(24, date_em=4)
		u_em = torch.cat([u_em1,u_em2,u_em3],dim=-1) # Shape: (batch, 209, 12)

		# Edge connection
		for i in range(self.group_num):
			for j in range(self.group_num):
				if i == j: continue

				# ReLU(enc(Z_i, Z_j, T))
				g_edge_input = torch.cat([g_x[:,i],g_x[:,j],u_em],dim=-1) # Shape: (batch, 44+44+12=100)
				tmp_g_edge_w = self.edge_inf(g_edge_input) # Shape: (batch, 12)

				tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0) # Shape: (1, batch, 209, 12)
				tmp_g_edge_index = torch.tensor([i,j]).unsqueeze(dim=0).to(self.device,non_blocking=True) # Shape: (1, 2)

				if i == 0 and j == 1:
					g_edge_w = tmp_g_edge_w # Shape: (1, batch, 12)
					g_edge_index = tmp_g_edge_index # Shape: (1, 2)
				else:
					g_edge_w = torch.cat([g_edge_w,tmp_g_edge_w],dim=0) # Shape: (210, batch, 12)
					g_edge_index = torch.cat([g_edge_index,tmp_g_edge_index],dim=0) # Shape: (210, 2)




		''' Group Message Passing
  			Group Update '''

		g_edge_w = g_edge_w.transpose(0,1)
		g_edge_index = g_edge_index.unsqueeze(dim=0)
		g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0],dim=0)
		g_edge_index = g_edge_index.transpose(1,2)
		g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)

		for i in range(self.gnn_layer):
			g_x = self.group_gnn[i](g_x,g_edge_index,g_edge_w)
		
		g_x = g_x.reshape(-1,self.group_num,g_x.shape[-1])
		


		''' City Group to City '''

		####################################################################################################	
		# S
		w2 = w.unsqueeze(dim=0)
		w2 = w2.repeat_interleave(g_x.size(0), dim=0)
		new_x = torch.bmm(w2, g_x) # Shape: (batch, 209, 32)


		if self.mode == 'both':
			# x Shape: (batch:64, 209, 32)
			x2 = x2.reshape(-1,x2.shape[-1])
			edge_w = edge_w.unsqueeze(dim=-1)

			for i in range(self.pred_step):
				
				""" Feedback """
				# x Shape: (batch, 209, 32)
				new_x = torch.cat([x, new_x], dim=-1) # Shape: (batch, 209, 64)


				""" City Update """
				tmp_edge_index = edge_index.clone()
				new_x, tmp_edge_w, tmp_edge_index = self.batchInput(new_x, edge_w, tmp_edge_index) # Shape: (batch*209, 64)

				for j in range(self.gnn_layer):
					new_x = self.global_gnn[j](new_x, tmp_edge_index, tmp_edge_w) # Shape: (batch*209, 32)

				""" Temporal """
				temp_x = torch.cat([x2, new_x],dim=-1)
				temp_x = self.TemporalAggregateMLP(temp_x)
				
				""" Final Forcasting """
				# new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, tmp_edge_index, tmp_edge_w)
				tmp_res = self.predMLP(temp_x) # Shape: (batch*209, 1)
				tmp_res = tmp_res.reshape(-1, self.city_num) # Shape: (batch, 209)
				tmp_res = tmp_res.unsqueeze(dim=-1) # Shape: (batch, 209, 1)
				if i == 0:
					res = tmp_res
				else:
					res = torch.cat([res,tmp_res],dim=-1) # Shape: (batch, 209, i+1)

				new_x = new_x.reshape(-1, self.city_num, self.x_em) # Shape: (batch, 209, 32)


		if self.mode == 'temp':
			""" City Update """
			new_x = torch.cat([x,new_x],dim=-1)
			edge_w = edge_w.unsqueeze(dim=-1)
			new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
			
			for i in range(self.gnn_layer):
				new_x = self.global_gnn[i](new_x,edge_index,edge_w)

			""" Temporal """
			x2 = x2.reshape(-1,x2.shape[-1])
			new_x = torch.cat([x2,new_x],dim=-1)
			new_x = self.TemporalAggregateMLP(new_x)

			""" Final Forcasting """
			new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w) # Shape: (batch*209, 32)
			res = self.predMLP(new_x)
			res = res.reshape(-1,self.city_num,self.pred_step) # Shape: (batch, 209, 6)



		if self.mode == 'feedback':
			edge_w = edge_w.unsqueeze(dim=-1)

			for i in range(self.pred_step):
				
				""" Feedback """
				# x Shape: (batch, 209, 32)
				new_x = torch.cat([x, new_x], dim=-1) # Shape: (batch, 209, 64)


				""" City Update """
				tmp_edge_index = edge_index.clone()
				new_x, tmp_edge_w, tmp_edge_index = self.batchInput(new_x, edge_w, tmp_edge_index) # Shape: (batch*209, 64)

				for j in range(self.gnn_layer):
					new_x = self.global_gnn[j](new_x, tmp_edge_index, tmp_edge_w) # Shape: (batch*209, 32)
				

				
				""" Final Forcasting """
				# new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, tmp_edge_index, tmp_edge_w)
				tmp_res = self.predMLP(new_x) # Shape: (batch*209, 1)
				tmp_res = tmp_res.reshape(-1, self.city_num) # Shape: (batch, 209)
				tmp_res = tmp_res.unsqueeze(dim=-1) # Shape: (batch, 209, 1)
				if i == 0:
					res = tmp_res
				else:
					res = torch.cat([res,tmp_res],dim=-1) # Shape: (batch, 209, i+1)

				new_x = new_x.reshape(-1, self.city_num, self.x_em) # Shape: (batch, 209, 32)

		
		if self.mode == 'test':
			""" City Update """
			new_x = torch.cat([x,new_x],dim=-1) # Shape: (batch, 209, 64) 
			edge_w = edge_w.unsqueeze(dim=-1)
			new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
			# new_x Shape: (batch, 209, 64) -> (batch*209, 64)
			# edge_w Shape: (batch, 4112) -> (batch*4112, 1)
			# edge_index Shape: (batch, 2, 4112) -> (2, batch*4112)

			for i in range(self.gnn_layer):
				new_x = self.global_gnn[i](new_x, edge_index, edge_w) # Shape: (batch*209, 32)


			""" Final Forcasting """
			outputs = new_x.unsqueeze(dim=0) # Shape: (1, batch*209, 32)

			for i in range(self.pred_step):
				new_x_attn = self.tmp_encoder_layer(outputs) # Shape: (i+1, batch*209, 32)
				new_x_attn = new_x_attn[-1] # Shape: (batch*209, 32)

				tmp_res = self.predMLP(new_x_attn) # Shape: (batch*209, 1)
				tmp_res = tmp_res.reshape(-1, self.city_num) # Shape: (batch, 209)
				tmp_res = tmp_res.unsqueeze(dim=-1) # Shape: (batch, 209, 1)
				if i == 0:
					res = tmp_res
				else:
					res = torch.cat([res,tmp_res],dim=-1) # Shape: (batch, 209, i+1)

				new_x_attn = new_x_attn.unsqueeze(0)
				outputs = torch.cat([outputs, new_x_attn], dim=0) # Shape: (i+2, batch*209, 32)


		if self.mode == 'ag':
			""" City Update """
			new_x = torch.cat([x,new_x],dim=-1) # Shape: (batch:64, 209, 64) 
			edge_w = edge_w.unsqueeze(dim=-1)
			new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)

			for i in range(self.gnn_layer):
				new_x = self.global_gnn[i](new_x,edge_index,edge_w)


			""" Final Forcasting """
			for i in range(self.pred_step):
				new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
				tmp_res = self.predMLP(new_x)
				tmp_res = tmp_res.reshape(-1,self.city_num)
				tmp_res = tmp_res.unsqueeze(dim=-1)
				if i == 0:
					res = tmp_res
				else:
					res = torch.cat([res,tmp_res],dim=-1)


		
		if self.mode == 'full':
			""" City Update """
			new_x = torch.cat([x,new_x],dim=-1) # Shape: (batch, 209, 64) 
			edge_w = edge_w.unsqueeze(dim=-1)
			new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
			# new_x Shape: (batch, 209, 64) -> (batch*209, 64)
			# edge_w Shape: (batch, 4112) -> (batch*4112, 1)
			# edge_index Shape: (batch, 2, 4112) -> (2, batch*4112)

			for i in range(self.gnn_layer):
				new_x = self.global_gnn[i](new_x, edge_index, edge_w) # Shape: (batch*209, 32)


			""" Final Forcasting """
			new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w) # Shape: (batch*209, 32)
			res = self.predMLP(new_x)
			res = res.reshape(-1,self.city_num,self.pred_step) # Shape: (batch, 209, 6)

		####################################################################################################	

		return res

class DecoderModule(nn.Module):
	def __init__(self,x_em,edge_h,gnn_h,gnn_layer,city_num,group_num,device):
		super(DecoderModule, self).__init__()
		self.device = device
		self.city_num = city_num
		self.group_num = group_num
		self.gnn_layer = gnn_layer
		self.x_embed = Lin(gnn_h, x_em)
		self.group_gnn = nn.ModuleList([NodeModel(x_em,edge_h,gnn_h)])
		for i in range(self.gnn_layer-1):
			self.group_gnn.append(NodeModel(gnn_h,edge_h,gnn_h))
		self.global_gnn = nn.ModuleList([NodeModel(x_em+gnn_h,1,gnn_h)])
		for i in range(self.gnn_layer-1):
			self.global_gnn.append(NodeModel(gnn_h,1,gnn_h))

	def forward(self, x, trans_w, g_edge_index, g_edge_w, edge_index, edge_w):
		x = self.x_embed(x)
		x = x.reshape(-1,self.city_num,x.shape[-1])

		# S
		w = Parameter(trans_w,requires_grad=False).to(self.device,non_blocking=True)
		w1 = w.transpose(0,1)
		w1 = w1.unsqueeze(dim=0)
		w1 = w1.repeat_interleave(x.size(0), dim=0)
		g_x = torch.bmm(w1, x)
		g_x = g_x.reshape(-1, g_x.shape[-1])


		for i in range(self.gnn_layer):
			g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)
		g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])

		# S
		w2 = w.unsqueeze(dim=0)
		w2 = w2.repeat_interleave(g_x.size(0), dim=0)
		new_x = torch.bmm(w2,g_x)

		# H
		new_x = torch.cat([x,new_x],dim=-1)
		new_x = new_x.reshape(-1,new_x.shape[-1])

		for i in range(self.gnn_layer):
			new_x = self.global_gnn[i](new_x,edge_index,edge_w)

		return new_x


class NodeModel(torch.nn.Module):
    def __init__(self,node_h,edge_h,gnn_h):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h+edge_h,gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h+gnn_h,gnn_h), ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)