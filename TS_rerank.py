from numpy.random import beta as betaDist

# Example of reranking a set of query results, with scores, using Thompson Sampling with informative priors
def rerank(self, results, user_id, user_feedback_map):
      #Reranking the results using thompson sampling
      yn_score = [x['y_score']/((x['y_score'] + x['n_score'])) for x in results if x['y_score'] + x['n_score'] > 0]
      # Only docs with feedback count as evidence for the prior
      avg_yn = sum(yn_score)/(len(yn_score) or 1)
      # 1,1 uninformative priors
      p_alpha = (avg_yn * len(yn_score)) + 1
      p_beta = ((1.0 - avg_yn) * len(yn_score)) + 1
      for doc in results:
          #Update beta with user individual feedbacks
          actual_y, actual_n = user_feedback_map[user_id+':'+doc['id']] or (0,0)
          doc['beta_score'] = betaDist(p_alpha + actual_y, p_beta + actual_n)
      # the +1 puts an upperbound on the beta influence. It never surpasses a doubled score.
      return sorted(results, key=lambda doc: doc['score']*(doc['beta_score']), reverse=True)
