import os
import hashlib
import math
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
from utils.prepare4llm import get_desc

warnings.filterwarnings('ignore')


class SourceTextPerturber:
    def __init__(self, ratio=0.0, targets=None, seed=2025,
                 intrinsic_pool=None, future_pool=None, split_name="train"):
        self.ratio = max(0.0, float(ratio or 0.0))
        if self.ratio > 1.0 and self.ratio <= 100.0:
            self.ratio /= 100.0
        self.targets = self._parse_targets(targets)
        self.seed = int(seed if seed is not None else 2025)
        self.split_name = split_name
        intrinsic_pool = self._clean_pool(intrinsic_pool)
        future_pool = self._clean_pool(future_pool)
        all_pool = intrinsic_pool + future_pool
        self.intrinsic_source_size = len(intrinsic_pool)
        self.future_source_size = len(future_pool)
        self.pools = {
            "intrinsic": intrinsic_pool + future_pool,
            "future": future_pool + intrinsic_pool,
        }
        self.all_pool = all_pool

    @staticmethod
    def _parse_targets(targets):
        if targets is None:
            return {"intrinsic", "future"}
        if isinstance(targets, str):
            items = targets.replace(";", ",").split(",")
        else:
            items = list(targets)
        out = {str(item).strip().lower() for item in items if str(item).strip()}
        aliases = {"scenario": "future", "description": "future", "coarse": "future"}
        out = {aliases.get(item, item) for item in out}
        return {item for item in out if item in {"intrinsic", "future"}}

    @staticmethod
    def _clean_pool(texts):
        if texts is None:
            return []
        out = []
        for text in texts:
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if text:
                out.append(text)
        return out

    @property
    def enabled(self):
        return self.ratio > 0 and bool(self.targets) and bool(self.all_pool)

    def _rng(self, *parts):
        payload = "|".join(str(part) for part in (self.seed, self.split_name, *parts))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return random.Random(int.from_bytes(digest, byteorder="little", signed=False))

    def _noise_words(self, rng, target, n_words, original_text):
        pool = self.pools.get(target) or self.all_pool
        if not pool:
            return []
        words = []
        attempts = 0
        max_attempts = max(10, n_words * 4)
        while len(words) < n_words and attempts < max_attempts:
            attempts += 1
            candidate = rng.choice(pool)
            if candidate == original_text and len(pool) > 1:
                continue
            cand_words = candidate.split()
            if not cand_words:
                continue
            need = n_words - len(words)
            if len(cand_words) > need:
                start = rng.randrange(0, len(cand_words) - need + 1)
                cand_words = cand_words[start:start + need]
            words.extend(cand_words)
        return words[:n_words]

    def perturb(self, text, target, sample_index, text_index=0):
        if not self.enabled or target not in self.targets:
            return text
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = " ".join(text.split())
        words = text.split()
        if not words:
            return text
        n_noise = max(1, int(math.ceil(len(words) * self.ratio)))
        rng = self._rng(target, sample_index, text_index, text)
        noise_words = self._noise_words(rng, target, n_noise, text)
        if not noise_words:
            return text
        insert_at = rng.randrange(0, len(words) + 1)
        return " ".join(words[:insert_at] + noise_words + words[insert_at:])


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 text_len=1, scaler_type='standard',
                 use_intrinsic=True, use_future_context=True,
                 text_perturb_ratio=0.0, text_perturb_targets=None,
                 text_perturb_seed=2025):
        
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.text_len = text_len
        self.use_intrinsic = use_intrinsic
        self.use_future_context = use_future_context
        self.flag = flag
        self.text_perturb_ratio = float(text_perturb_ratio or 0.0)
        self.text_perturb_targets = text_perturb_targets
        self.text_perturb_seed = int(text_perturb_seed if text_perturb_seed is not None else 2025)

        self.root_path = root_path
        self.data_path = data_path
        self.data_prefix = data_path.split('.')[0]
        
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()
        self.domain = data_path.split('/')[0]
        self.desc = get_desc(self.domain, self.seq_len, self.pred_len)
        self.text_perturber = self._build_text_perturber()
        if self.text_perturber.enabled:
            print(
                f"[TEXT_PERTURB][{self.flag}] ratio={self.text_perturber.ratio:.2f} "
                f"targets={sorted(self.text_perturber.targets)} "
                f"intrinsic_source={self.text_perturber.intrinsic_source_size} "
                f"future_source={self.text_perturber.future_source_size}"
            )
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def _build_text_perturber(self):
        intrinsic_pool = []
        future_pool = []
        if getattr(self, "txt_it", None) is not None:
            for col in ("intrinsic", "trigger"):
                if col in self.txt_it.columns:
                    intrinsic_pool.extend(self.txt_it[col].fillna("").astype(str).tolist())
        if getattr(self, "txt_cp", None) is not None and "coarse_pred" in self.txt_cp.columns:
            future_pool.extend(self.txt_cp["coarse_pred"].fillna("").astype(str).tolist())
        return SourceTextPerturber(
            ratio=self.text_perturb_ratio,
            targets=self.text_perturb_targets,
            seed=self.text_perturb_seed,
            intrinsic_pool=intrinsic_pool,
            future_pool=future_pool,
            split_name=self.flag,
        )

    def _perturb_text(self, text, target, sample_index, text_index=0):
        perturber = getattr(self, "text_perturber", None)
        if perturber is None:
            return text
        return perturber.perturb(text, target, sample_index, text_index)

    def __read_data__(self):
        df_num = pd.read_csv(os.path.join(self.root_path, 'numerical', self.data_path))
        
        
        
        
        it_path = os.path.join(self.root_path, 'textual', self.data_prefix + '_intrinsic_trigger.csv')
        cp_path = os.path.join(self.root_path, 'textual', self.data_prefix + '_coarse_pred.csv')
        self.has_it = os.path.exists(it_path)
        self.has_cp = os.path.exists(cp_path)
        ap_path = os.path.join(self.root_path, 'textual', self.data_prefix + '_abnormal_points.jsonl')
        self.has_ap = os.path.exists(ap_path)
        if self.has_ap:
            df_ap = pd.read_json(ap_path, lines=True)
            df_ap['end_date'] = pd.to_datetime(df_ap['end_date'])
            self.txt_ap = df_ap
        else:
            self.txt_ap = None
        if self.has_it:
            df_it = pd.read_csv(it_path)
        if self.has_cp:
            df_cp = pd.read_csv(cp_path)

        df_num = df_num.dropna(axis='index', how='any', subset=['OT'])

        df_num['date'], df_num['start_date'], df_num['end_date'] = pd.to_datetime(df_num['date']), pd.to_datetime(df_num['start_date']), pd.to_datetime(df_num['end_date'])

        df_num = df_num.sort_values('date', ascending=True).reset_index(drop=True)
        num_train = int(len(df_num) * 0.7)
        num_test = int(len(df_num) * 0.2)
        num_vali = len(df_num) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_num) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_num)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        
        first_start_date = df_num.start_date[border1]
        final_end_date = df_num.end_date[border2-1]

        if self.has_ap:
            self.txt_ap = self.txt_ap.loc[
                (self.txt_ap.end_date >= first_start_date) & (self.txt_ap.end_date <= final_end_date)
            ].reset_index(drop=True)

        df_data = df_num[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
            self.mean_data = self.scaler.mean_
            self.std_data = self.scaler.scale_
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_num[['date']][border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


        self.data_stamp = data_stamp
        self.num_dates = df_num[['start_date', 'end_date']][border1:border2].reset_index(drop=True)

        if self.has_it:
            df_it['end_date'] = pd.to_datetime(df_it['end_date'])
            self.txt_it = df_it.loc[(df_it.end_date >= first_start_date) & (df_it.end_date <= final_end_date)].reset_index(drop=True)
        else:
            self.txt_it = None

        if self.has_cp:
            df_cp['end_date'] = pd.to_datetime(df_cp['end_date'])
            self.txt_cp = df_cp.loc[(df_cp.end_date >= first_start_date) & (df_cp.end_date <= final_end_date)].reset_index(drop=True)
        else:
            self.txt_cp = None

    def collect_intrinsic_texts(self, s_begin_idx, s_end_idx):
        """Internal helper."""
        if self.txt_it is None:
            return [""] * (s_end_idx - s_begin_idx)
        
        dates = self.num_dates.end_date[s_begin_idx:s_end_idx].reset_index(drop=True)
        merged = pd.merge(
            pd.DataFrame({"end_date": dates}),
            self.txt_it[["end_date","intrinsic"]],
            on="end_date",
            how="left"
        )
        texts = merged["intrinsic"].fillna("").astype(str).str.strip().tolist()
        return [
            self._perturb_text(text, "intrinsic", s_begin_idx, offset)
            for offset, text in enumerate(texts)
        ]

    def collect_future_points(self, s_end_idx):
        if self.txt_ap is None:
            return ""
        end_date = self.num_dates.end_date[s_end_idx - 1]
        row = self.txt_ap.loc[self.txt_ap.end_date == end_date]
        if row.empty:
            return ""
        pts = row.iloc[0].get("points", [])
        import json
        return json.dumps({"points": pts}, ensure_ascii=False)

    def collect_future_context(self, s_end_idx, sample_index=None):
        """Internal helper."""
        if self.txt_cp is None:
            return ""
        end_date = self.num_dates.end_date[s_end_idx-1]
        row = self.txt_cp.loc[self.txt_cp.end_date == end_date]

        text = (row.iloc[0].coarse_pred if not row.empty else "")
        return self._perturb_text(text, "future", s_end_idx if sample_index is None else sample_index, 0)


    def collect_text(self, start_date, end_date):
        report = self.txt_report.loc[(self.txt_report.end_date >= start_date) & (self.txt_report.end_date <= end_date)]
        def add_datemark(row):
            return row['start_date'].strftime("%Y-%m-%d") + " to " + row['end_date'].strftime("%Y-%m-%d") + ": " + row['fact']
        if not report.empty:
            report = report.apply(add_datemark, axis=1).to_list()
            report.insert(0, self.desc)
            text_mark = 1
        else:
            report = ['NA']
            text_mark = 0
        all_txt = ' '.join(report)
        return all_txt, text_mark
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
    
        seq_x_stamp = self.data_stamp[s_begin:s_end]
        seq_y_stamp = self.data_stamp[r_begin:r_end]

        step_texts = self.collect_intrinsic_texts(s_begin, s_end) if self.use_intrinsic else [""] * self.seq_len
        future_context = self.collect_future_context(s_end, index) if self.use_future_context else ""
        
        if self.use_future_context and future_context=="":
            print("No future context")

        future_points = self.collect_future_points(s_end)
        observed_data = np.concatenate([seq_x, seq_y], axis=0)
        timesteps = np.concatenate([seq_x_stamp, seq_y_stamp], axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate([np.ones_like(seq_x), np.zeros_like(seq_y)], axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len).astype(np.float32), 
            'feature_id': np.arange(seq_x.shape[1]).astype(np.float32),
            'timesteps': timesteps,
            'intrinsic_texts': step_texts,  
            'future_context': future_context,      
            'future_points': future_points,
        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
