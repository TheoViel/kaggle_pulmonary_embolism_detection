from torch.utils.data.sampler import BatchSampler


class PatientSampler(BatchSampler):
    """
    Custom PyTorch Sampler that limits the number of images per patient
    """

    def __init__(
        self, sampler, patients, batch_size=32, drop_last=False, samples_per_patient=10
    ):
        """
        Constructor.

        Args:
            sampler (torch sampler): Initial sampler for the dataset, e.g. RandomSampler
            patients ([type]): Patient corresponding to each sample. Precomputed to gain time.
            batch_size (int, optional): Batch size. Defaults to 32.
            drop_last (bool, optional): Whether to discard the last batch. Defaults to False.
            samples_per_patient (int, optional): Max of image to use per patient. Defaults to 10.
        """
        super().__init__(sampler, batch_size, drop_last)
        self.samples_per_patient = samples_per_patient
        self.patients = patients

        self.len = self.compute_len()

    def __len__(self):
        return self.len

    def compute_len(self):
        patient_counts = {}
        yielded = 0
        batch = []

        for idx in self.sampler:
            patient = self.patients[idx]
            try:
                patient_counts[patient] += 1
            except KeyError:
                patient_counts[patient] = 1

            if patient_counts[patient] <= self.samples_per_patient:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yielded += 1
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1

        return yielded

    def __iter__(self):
        """
        Iterator.
        Only adds an index to a batch if the associated patients has not be sampled too many time.

        Yields:
            torch tensors : batches.
        """
        patient_counts = {}
        yielded = 0
        batch = []

        for idx in self.sampler:
            patient = self.patients[idx]
            try:
                patient_counts[patient] += 1
            except KeyError:
                patient_counts[patient] = 1

            if patient_counts[patient] <= self.samples_per_patient:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    yielded += 1
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
