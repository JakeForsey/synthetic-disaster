import { Component, OnInit } from '@angular/core';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { ResultService } from 'src/app/result/result.service'

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit {

  resultService: ResultService;
  public result$: Observable<string>;

  constructor(resultService: ResultService) {
    this.resultService = resultService;
  }

  ngOnInit() {
  }

  onClick() {
    this.result$ = this.resultService.getResult()
  }
}
