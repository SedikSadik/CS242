import os
import nni
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_cifar
from model_factory import create_cnn_model, is_resnet
import tqdm

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    else:
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="TA Knowledge Distillation Code")
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--dataset",
        default="cifar100",
        type=str,
        help="dataset. can be either cifar10 or cifar100",
    )
    parser.add_argument(
        "--save_dir",
        default="model_save_dir",
        type=str,
        help="path to folder where models should be saved",
    )
    parser.add_argument(
        "--description",
        default="",
        type=str,
        help="description to differentiate different methods of training the same model size",
        required=True,
    )
    parser.add_argument("--batch-size", default=2048, type=int, help="batch_size")
    parser.add_argument(
        "--learning-rate", default=0.1, type=float, help="initial learning rate"
    )
    # IMPORTANT NEW ARGUMENT TO DECIDE ON WHAT METHOD IS BEST
    # PURAB TO DO. this will be transfered to the train_config of the student I think. use this to change the mode. maybe debug this if needed
    parser.add_argument(
        "--weighing_mode",
        default="equal_weight",
        type=float,
        help="weighing mode",
    )

    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        help="SGD weight decay (default: 1e-4)",
    )
    parser.add_argument("--teacher", default="", type=str, help="teacher student name")
    parser.add_argument(
        "--student", "--model", default="resnet8", type=str, help="teacher student name"
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="",
        type=str,
        help="optinal pretrained checkpoint for teacher",
    )
    parser.add_argument(
        "--cuda",
        default=False,
        type=str2bool,
        help="whether or not use cuda(train on GPU)",
    )
    parser.add_argument(
        "--dataset-dir", default="./data", type=str, help="dataset directory"
    )
    parser.add_argument("--ta", default="", type=str, help="ta stuff")
    parser.add_argument(
        "--ta-checkpoint", default="", type=str, help="ta checkpoint path"
    )
    parser.add_argument(
        "--student-checkpoint", default="", type=str, help="student checkpoint path"
    )
    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path, optimizer=None, get_epoch=False):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    if not os.path.isfile(checkpoint_path):
        print("-----------------------------")
        print("CHECKPOINT DOES NOT EXIST")
        print(checkpoint_path)
        print("-----------------------------")

    else:
        print("-----------------------------")
        print("CHECKPOINT DOES EXISTS")
        print(checkpoint_path)
        print("-----------------------------")
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp["model_state_dict"])
    if not get_epoch:
        return model
    else:
        print("LAODING OPTIMIZER")
        return model, model_ckp["optimizer_state_dict"], model_ckp["epoch"] + 1


class TrainManager(object):

    def __init__(
        self,
        student,
        teacher=None,
        ta=None,
        train_loader=None,
        test_loader=None,
        train_config={},
        optimizer_state_dict=None,
        start_epoch=None,
    ):
        self.student = student
        self.teacher = teacher
        self.ta = ta
        self.have_teacher = bool(self.teacher)
        self.have_ta = bool(self.ta)
        self.device = train_config["device"]
        self.name = train_config["name"]
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=train_config["learning_rate"],
            momentum=train_config["momentum"],
            weight_decay=train_config["weight_decay"],
        )
        if optimizer_state_dict:
            print("USING GIVEN OPTIMIZER!")
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.start_epoch = start_epoch if start_epoch else 0
        if self.have_teacher:
            self.teacher.eval()
            self.teacher.train(mode=False)

        if self.have_ta:
            self.ta.eval()
            self.ta.train(mode=False)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = train_config

        # assuming resnet:
        self.teacher_layers = int(self.teacher[6:])
        self.ta_layers = int(self.ta[6:])

        if train_config.get("weighing_mode"):

            if train_config["weighing_mode"] == "layer_proportional":
                self.get_teacher_ta_weights = self.teacher_ta_layer_proportion_weight
            # elif train_config["weighing_mode"] == "dynamic_correctness":
            else:
                self.get_teacher_ta_weights = self.teacher_ta_equal_weight
        else:
            self.get_teacher_ta_weights = self.teacher_ta_equal_weight

        self.get_teacher_ta_weights()

    def teacher_ta_equal_weight(
        self,
        hard_labels=None,
        teacher_out=None,
        ta_out=None,
        student_out=None,
        T=None,
    ) -> tuple[int, int]:
        return 0.5, 0.5

    def teacher_ta_layer_proportion_weight(
        self,
        hard_labels=None,
        teacher_out=None,
        ta_out=None,
        student_out=None,
        T=None,
    ) -> tuple[int, int]:
        total_layers = self.ta_layers + self.teacher_layers

        return self.ta_layers / total_layers, self.teacher_layers / total_layers

    # NOT DONE YET
    # def teacher_ta_dynamic_correctness_based_weight(
    #     self,
    #     hard_labels=None,
    #     teacher_out=None,
    #     ta_out=None,
    #     student_out=None,
    #     T=None,
    # ) -> tuple[int, int]:

    #     ce_loss = nn.CrossEntropyLoss()
    #     ce_teacher = F.cross_entropy(teacher_out, hard_labels)
    #     ce_ta = F.cross_entropy(ta_out, hard_labels)

    #     [conf_teacher, conf_ta] = F.softmax([-ce_teacher, -ce_ta])

    #     kl_teacher_ta = nn.KLDivLoss()(
    #         F.log_softmax(teacher_out, dim=1),
    #         ta_out,
    #     )

    #     kl_factor = F.sigmoid(kl_teacher_ta)

    #     return self.ta_layers / total_layers, self.teacher_layers / total_layers

    # def get_teacher_tf_weights(
    #     self,
    #     hard_labels=None,
    #     teacher_out=None,
    #     ta_out=None,
    #     student_out=None,
    #     T=None,
    #     teacher_layer_cnt=None,
    #     student_layer_cnt=None,
    # ) -> tuple[int, int]:

    def train(self):
        lambda_ = self.config["lambda_student"]
        T = self.config["T_student"]
        epochs = self.config["epochs"]
        trial_id = self.config["trial_id"]

        max_val_acc = 0
        iteration = 0
        best_acc = 0
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.start_epoch, epochs):
            self.student.train()
            self.adjust_learning_rate(self.optimizer, epoch)
            loss = 0
            for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.train_loader)):
                iteration += 1
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.student(data)
                # Standard Learning Loss ( Classification Loss)
                loss_SL = criterion(output, target)
                loss = loss_SL

                if self.have_teacher:
                    teacher_outputs = self.teacher(data)

                    teacher_soft = F.softmax(teacher_outputs / T, dim=1)

                    if self.have_ta:
                        ta_outputs = self.ta(data)
                        # Knowledge Distillation Loss
                        ta_soft = F.softmax(ta_outputs / T, dim=1)

                        w_teacher, w_ta = self.get_teacher_ta_weights(
                            target,
                            teacher_outputs,
                            ta_outputs,
                            output,
                            T,
                        )

                        combined_outputs = teacher_soft * w_teacher + ta_soft * w_ta

                    else:
                        combined_outputs = teacher_soft

                    loss_KD = nn.KLDivLoss()(
                        F.log_softmax(output / T, dim=1),
                        combined_outputs,
                    )
                    loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD

                loss.backward()
                self.optimizer.step()

            print("epoch {}/{}".format(epoch, epochs))
            val_acc = self.validate(step=epoch)
            description = self.config["args"].description.replace(" ", "_")
            if epoch % 10 == 0:
                self.save(
                    epoch,
                    name=f"{self.name}_epoch{epoch}_acc_{val_acc}_{description}.pth.tar",
                )

            if val_acc > best_acc:
                best_acc = val_acc
                self.save(
                    epoch,
                    name=f"{self.name}_best_at_epoch{epoch}_acc_{val_acc}_{description}.pth.tar",
                )
                # self.save(
                #     epoch,
                #     name=f"{self.name}_{trial_id}_{description}_best.pth.tar",
                # )
                self.save(
                    epoch,
                    name=f"{self.name}_{description}_best.pth.tar",
                )

        return best_acc

    def validate(self, step=0):
        self.student.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # self.accuracy_history.append(acc)
            acc = 100 * correct / total

            print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
            return acc

    def save(self, epoch, name=None):
        trial_id = self.config["trial_id"]
        save_dir = self.config["args"].save_dir
        description = self.config["args"].description.replace(" ", "_")

        if name is None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.student.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(
                    save_dir, f"{self.name}_epoch{epoch}_{description}.pth.tar"
                ),
            )
        else:
            torch.save(
                {
                    "model_state_dict": self.student.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(save_dir, name),
            )

    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config["epochs"]
        models_are_plane = self.config["is_plane"]

        # depending on dataset
        if models_are_plane:
            lr = 0.01
        else:
            if epoch < int(epoch / 2.0):
                lr = 0.1
            elif epoch < int(epochs * 3 / 4.0):
                lr = 0.1 * 0.1
            else:
                lr = 0.1 * 0.01

        # update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    # Parsing arguments and prepare settings for training
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    config = nni.get_next_parameter()
    print("config: ", config)
    config = {
        "seed": 42,
        "T_student": 5,
        "lambda_student": 0.5,
    }
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    dataset = args.dataset
    num_classes = 100 if dataset == "cifar100" else "cifar10"
    teacher_model = None
    ta_model = None

    optimizer = None
    student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)
    start_epoch = 0
    optimizer_state_dict = None
    if args.student_checkpoint:
        student_model, optimizer_state_dict, start_epoch = load_checkpoint(
            student_model, args.student_checkpoint, get_epoch=True
        )

    train_config = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "device": "cuda" if args.cuda else "cpu",
        "is_plane": not is_resnet(args.student),
        "trial_id": trial_id,
        "T_student": config.get("T_student"),
        "lambda_student": config.get("lambda_student"),
        "args": args,
        "student_start_epoch": start_epoch if start_epoch else None,
    }

    # Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
    # This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
    if args.teacher:
        teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
        if args.teacher_checkpoint:
            print("---------- Loading Teacher -------")
            teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
        else:
            print("---------- Training Teacher -------")
            train_loader, test_loader = get_cifar(
                num_classes, batch_size=args.batch_size
            )
            teacher_train_config = copy.deepcopy(train_config)
            teacher_name = "{}_{}_best.pth.tar".format(args.teacher, trial_id)
            teacher_train_config["name"] = args.teacher
            teacher_trainer = TrainManager(
                teacher_model,
                teacher=None,
                train_loader=train_loader,
                test_loader=test_loader,
                train_config=teacher_train_config,
            )
            description = args.description.replace(" ", "_")
            teacher_trainer.train()
            checkpoint_path = f"{args.teacher}_{trial_id}_{description}_best.pth.tar"
            teacher_model = load_checkpoint(teacher_model, checkpoint_path)

    # TA model
    if args.ta:
        print(args.ta, "WTF")
        ta_model = create_cnn_model(args.ta, dataset, use_cuda=args.cuda)
        if args.ta_checkpoint:
            print("---------- Loading Teacher Assistant -------")
            ta_model = load_checkpoint(ta_model, args.ta_checkpoint)
        else:
            print("---------- Training Teacher Assistant -------")
            train_loader, test_loader = get_cifar(
                num_classes, batch_size=args.batch_size
            )
            ta_train_config = copy.deepcopy(train_config)
            ta_name = "{}_{}_best.pth.tar".format(args.ta, trial_id)
            ta_train_config["name"] = args.ta
            ta_trainer = TrainManager(
                ta_model,
                teacher=teacher_model,
                train_loader=train_loader,
                test_loader=test_loader,
                train_config=teacher_train_config,
            )
            description = args.description.replace(" ", "_")
            ta_trainer.train()
            checkpoint_path = f"{args.ta}_{trial_id}_{description}_best.pth.tar"

            ta_model = load_checkpoint(ta_model, checkpoint_path)

    # Student training
    print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    train_loader, test_loader = get_cifar(num_classes, batch_size=args.batch_size)
    student_train_config["name"] = args.student
    student_trainer = TrainManager(
        student_model,
        teacher=teacher_model,
        ta=ta_model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_config=student_train_config,
        optimizer_state_dict=optimizer_state_dict,
        start_epoch=start_epoch,
    )
    best_student_acc = student_trainer.train()
    nni.report_final_result(best_student_acc)
