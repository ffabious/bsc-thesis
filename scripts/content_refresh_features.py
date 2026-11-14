"""Four strictly historical content cues from a frozen recommender item table."""

import numpy as np

from refresh_controller import FEATURES,HistoryFeatures

CONTENT_FEATURES=FEATURES+["latest_matches_cached","latest_in_cached50","latest_cached_cosine","latest_log_train_count"]


class ContentFeatures(HistoryFeatures):
    def __init__(self,stream,prefix_users,prefix_times,embeddings,train_counts):
        super().__init__(stream,prefix_users,prefix_times)
        self.embeddings=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True).clip(1e-12)
        self.log_counts=np.log1p(train_counts);self.latest=stream["queries"][:,-1]
        self.static_cpu_bytes=self.embeddings.nbytes+self.log_counts.nbytes

    def row(self,i,previous):
        current=self.latest[i];cached=self.stream["queries"][previous];old=cached[-1]
        extra=[current==old,bool((cached==current).any()),self.embeddings[current]@self.embeddings[old],self.log_counts[current]]
        return np.r_[super().row(i,previous),extra]

    def matrix(self,requests,previous):
        extra=np.empty((len(requests),4))
        for start in range(0,len(requests),8192):
            take=slice(start,start+8192);current=self.latest[requests[take]];cached=self.stream["queries"][previous[take]];old=cached[:,-1]
            extra[take]=np.column_stack([current==old,(cached==current[:,None]).any(1),
                np.einsum("ij,ij->i",self.embeddings[current],self.embeddings[old]),self.log_counts[current]])
        return np.column_stack([super().matrix(requests,previous),extra])
