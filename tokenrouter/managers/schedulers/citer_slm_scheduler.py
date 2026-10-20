from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.algorithms.citer.conf_model import ConfidenceModel, resolve_router_path
from tokenrouter.managers.peer_request import PeerReq, SimpleSamplingParams
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class CITERSLMScheduler(BaseTokenRoutingScheduler):

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta,
        )
        scheduler_kwargs = server_args.scheduler_kwargs
        self.threshold = scheduler_kwargs.pop("threshold", 0.5)
        self.citer_router = ConfidenceModel(
            self.model_config,
            self.threshold,
            self.device,
        )
        router_path = resolve_router_path(scheduler_kwargs["router_path"])
        self.citer_router.load_state_dict(
            torch.load(router_path, map_location=self.device)
        )
        self.citer_router.eval()

    def route(self, batch, result) -> torch.Tensor:
        device = batch.seq_lens.device
        batch_size = batch.batch_size()
        is_prefill = result.logits_output.hidden_states.shape[0] != batch_size

        if is_prefill:
            # For prefill, use cumsum of extend_lens to get correct indices
            extend_lens = torch.tensor(batch.extend_lens, device=device)
            hidden_indices = torch.cumsum(extend_lens, dim=0) - 1
        else:
            # For decode, use sequential indices
            hidden_indices = torch.arange(batch_size, device=device)

        hidden_states = result.logits_output.hidden_states[hidden_indices, :][
            :, None, :
        ]  # batch_size, 1, hidden_size

        return self.citer_router(hidden_states).reshape(batch_size).cpu()

    def send(self, req: Req, destination: Optional[str] = None) -> PeerReq:
        destination = destination or self.peer_name
        req.model_tokens[self.server_args.server_name][1] -= 1
        current_token_ids = req.origin_input_ids + req.output_ids[:-1]
        req_to_send = PeerReq.construct_from_req(
            req=req,
            source=self.server_args.server_name,
            destination=destination,
            current_token_ids=current_token_ids,
            entry_model=self.server_args.entry_model_name,
            handoff_seq=req.handoff_seq,
        )
        return req_to_send
