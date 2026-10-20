from typing import Optional

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.utils import DPBalanceMeta

from tokenrouter.algorithms.r2r.dataclass import ModelOutputs
from tokenrouter.algorithms.r2r.switching import create_switching_strategy
from tokenrouter.managers.peer_request import PeerReq
from tokenrouter.managers.schedulers.base_scheduler import BaseTokenRoutingScheduler
from tokenrouter.server_args import PortArgs, ServerArgs


class R2RSLMScheduler(BaseTokenRoutingScheduler):

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
        self.strategy_name = scheduler_kwargs.pop("switching_strategy", "neural")
        self.r2r_router = create_switching_strategy(
            self.strategy_name, **scheduler_kwargs
        )

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

        # Get hidden states for the relevant positions
        hidden_states = result.logits_output.hidden_states[hidden_indices, :][
            :, None, :
        ]  # batch_size, 1, hidden_size
        logits = (result.logits_output.next_token_logits)[
            :, None, :
        ]  # batch_size, 1, vocab_size
        next_token_ids = result.next_token_ids  # batch_size, 1

        model_outputs = ModelOutputs(
            hidden_states=[hidden_states], logits=logits, token=next_token_ids[:, None]
        )
        if self.strategy_name == "momentum":
            return self.r2r_router.route(model_outputs, [req.rid for req in batch.reqs]).cpu()
        return self.r2r_router.route(model_outputs).cpu()

    def _mark_terminal(self, rid: str, handoff_seq: int) -> None:
        super()._mark_terminal(rid, handoff_seq)
        if self.strategy_name == "momentum":
            self.r2r_router.request_states.pop(rid, None)

    def abort_request(self, recv_req):
        super().abort_request(recv_req)
        if self.strategy_name == "momentum":
            for rid in list(self.r2r_router.request_states):
                if recv_req.abort_all or rid.startswith(recv_req.rid):
                    self.r2r_router.request_states.pop(rid, None)

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
